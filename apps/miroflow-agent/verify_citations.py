#!/usr/bin/env python3
# Copyright 2025 Miromind.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Citation Verification Script

This script verifies that citations in a report actually match the content
from the source documents (long_context.json and local files).

It uses LLM as a judge to determine if the cited claim is supported by the source.

Usage:
    # Verify a single report
    python verify_citations.py --report results/bench_case1104/001.md --data data/bench_case1104/001
    
    # Verify with verbose output
    python verify_citations.py --report results/bench_case1104/001.md --data data/bench_case1104/001 --verbose
"""

import os
import sys
import re
import json
import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation reference in the report."""
    citation_id: str
    citation_type: str  # "RAG" or "Doc"
    title: Optional[str] = None
    chunk_indices: Optional[List[int]] = None
    filename: Optional[str] = None


@dataclass
class CitedClaim:
    """A claim in the report with its citations."""
    text: str
    citations: List[Citation]


@dataclass
class VerificationResult:
    """Result of verifying a citation."""
    claim_text: str
    citation_id: str
    source_content: Optional[str]
    is_supported: bool
    confidence: str  # "high", "medium", "low"
    explanation: str
    source_found: bool = True


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching (handle traditional/simplified Chinese, etc.)."""
    # Remove common variations
    title = title.strip()
    # Remove trailing ellipsis
    title = re.sub(r'\.{3,}$', '', title)
    title = re.sub(r'…$', '', title)
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title)
    return title


def fuzzy_match_title(citation_title: str, db_titles: List[str]) -> Optional[str]:
    """Find the best matching title from database using fuzzy matching."""
    citation_norm = normalize_title(citation_title)
    
    # Try exact match first
    for db_title in db_titles:
        if normalize_title(db_title) == citation_norm:
            return db_title
    
    # Try prefix match (for truncated titles with ...)
    for db_title in db_titles:
        db_norm = normalize_title(db_title)
        # Check if one is a prefix of the other
        if db_norm.startswith(citation_norm[:20]) or citation_norm.startswith(db_norm[:20]):
            return db_title
    
    # Try substring match
    for db_title in db_titles:
        db_norm = normalize_title(db_title)
        # Check significant overlap
        if len(citation_norm) > 10 and len(db_norm) > 10:
            # Check if first 15 chars match (ignoring minor differences)
            if citation_norm[:15] == db_norm[:15]:
                return db_title
    
    # Try character-level similarity for Chinese text
    best_match = None
    best_score = 0
    for db_title in db_titles:
        db_norm = normalize_title(db_title)
        # Count matching characters
        common_chars = set(citation_norm) & set(db_norm)
        score = len(common_chars) / max(len(set(citation_norm)), len(set(db_norm)))
        if score > best_score and score > 0.7:  # At least 70% character overlap
            best_score = score
            best_match = db_title
    
    return best_match


def parse_citations_section(report_text: str) -> Dict[str, Citation]:
    """Parse the Citations section at the end of the report."""
    citations = {}
    
    # Find the Citations section (try multiple patterns)
    patterns = [
        r'\*\*Citations:\*\*\s*\n(.*?)(?:\n\n|\Z)',
        r'\\textbf\{Citations:\}\s*\n(.*?)(?:\n\n|\Z)',
        r'Citations:\s*\n(.*?)(?:\n\n|\Z)',
    ]
    
    citations_text = None
    for pattern in patterns:
        match = re.search(pattern, report_text, re.DOTALL)
        if match:
            citations_text = match.group(1)
            break
    
    if not citations_text:
        logger.warning("No Citations section found in report")
        return citations
    
    # Parse each citation line
    for line in citations_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Parse RAG citations: [RAG-1]: 标题, chunk 0,1,2
        rag_match = re.match(r'\[RAG-(\d+)\]:\s*(.+?),\s*chunk\s*([\d,\s]+)', line)
        if rag_match:
            rag_id = f"RAG-{rag_match.group(1)}"
            title = rag_match.group(2).strip()
            chunk_str = rag_match.group(3)
            chunk_indices = [int(c.strip()) for c in chunk_str.split(',')]
            citations[rag_id] = Citation(
                citation_id=rag_id,
                citation_type="RAG",
                title=title,
                chunk_indices=chunk_indices
            )
            continue
        
        # Parse Doc citations: [Doc: filename]: filename
        doc_match = re.match(r'\[Doc:\s*(.+?)\]:\s*(.+)', line)
        if doc_match:
            filename = doc_match.group(1).strip()
            doc_id = f"Doc: {filename}"
            citations[doc_id] = Citation(
                citation_id=doc_id,
                citation_type="Doc",
                filename=filename
            )
    
    return citations


def extract_cited_claims(report_text: str) -> List[CitedClaim]:
    """Extract sentences/claims with citations from the report."""
    claims = []
    
    # Find the Final Report section (stop before Citations)
    report_match = re.search(r'## Final Report\s*\n(.*?)(?=\\vspace|\\textbf\{Citations|\*\*Citations|\n## |\Z)', report_text, re.DOTALL)
    if not report_match:
        report_match = re.search(r'\\boxed\{(.*?)(?=\\vspace|\\textbf\{Citations)', report_text, re.DOTALL)
    
    if not report_match:
        logger.warning("Could not find report content")
        return claims
    
    report_content = report_match.group(1)
    
    # Split into sentences and find those with citations
    citation_pattern = r'\[(?:RAG-\d+|Doc:\s*[^\]]+)\]'
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?。！？])\s+', report_content)
    
    for sentence in sentences:
        citation_matches = re.findall(citation_pattern, sentence)
        if citation_matches:
            sentence_citations = []
            for match in citation_matches:
                if match.startswith('[RAG-'):
                    rag_id = re.search(r'RAG-\d+', match).group(0)
                    sentence_citations.append(Citation(citation_id=rag_id, citation_type="RAG"))
                elif match.startswith('[Doc:'):
                    filename = re.search(r'Doc:\s*([^\]]+)', match).group(1).strip()
                    sentence_citations.append(Citation(
                        citation_id=f"Doc: {filename}",
                        citation_type="Doc",
                        filename=filename
                    ))
            
            if sentence_citations:
                # Clean the sentence text
                clean_sentence = re.sub(citation_pattern, '', sentence).strip()
                claims.append(CitedClaim(text=clean_sentence, citations=sentence_citations))
    
    return claims


def load_chunks_from_db(db_path: str) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    """Load chunks from the preprocessed database. Returns (chunks_dict, all_titles)."""
    if not os.path.exists(db_path):
        return {}, []
    
    chunks = {}
    titles = set()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT title, chunk_index, content FROM chunks")
        for title, chunk_index, content in cursor.fetchall():
            chunks[(title, chunk_index)] = content
            titles.add(title)
    except sqlite3.OperationalError:
        logger.warning(f"Could not read chunks from {db_path}")
    finally:
        conn.close()
    
    return chunks, list(titles)


def find_source_content(
    citation: Citation,
    citation_info: Dict[str, Citation],
    chunks: Dict[Tuple[str, int], str],
    all_titles: List[str],
    data_dir: str
) -> Tuple[Optional[str], bool]:
    """Find the source content for a citation. Returns (content, found)."""
    full_citation = citation_info.get(citation.citation_id)
    
    if citation.citation_type == "RAG":
        if not full_citation or not full_citation.title:
            return None, False
        
        # Try to find matching title using fuzzy matching
        matched_title = fuzzy_match_title(full_citation.title, all_titles)
        
        if not matched_title:
            logger.warning(f"Could not find matching title for: {full_citation.title}")
            return None, False
        
        # Find chunks by matched title and index
        contents = []
        for chunk_idx in full_citation.chunk_indices or []:
            key = (matched_title, chunk_idx)
            if key in chunks:
                contents.append(chunks[key])
            else:
                logger.warning(f"Chunk not found: {matched_title}, index {chunk_idx}")
        
        return ("\n\n".join(contents) if contents else None, len(contents) > 0)
    
    elif citation.citation_type == "Doc":
        filename = citation.filename or (full_citation.filename if full_citation else None)
        if filename:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                return f"[File exists: {filename}]", True
    
    return None, False


def verify_with_llm(
    claim_text: str,
    source_content: str,
    client: OpenAI
) -> Tuple[bool, str, str]:
    """Use LLM to verify if the claim is supported by the source."""
    
    prompt = f"""You are a citation verification expert. Your task is to determine if a claim from a report is supported by the cited source content.

## Claim from Report:
{claim_text}

## Cited Source Content:
{source_content}

## Instructions:
1. Analyze if the claim can be reasonably derived from or supported by the source content.
2. Consider that the claim may be a paraphrase, translation, or summary of the source.
3. The claim may be in English while the source is in Chinese - this is expected.
4. Be lenient - if the core information matches, consider it supported.

## Response Format (JSON):
{{
    "is_supported": true/false,
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of your judgment (in English)"
}}

Respond with only the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        # Parse JSON response
        result = json.loads(result_text)
        return (
            result.get("is_supported", False),
            result.get("confidence", "low"),
            result.get("explanation", "No explanation provided")
        )
    except Exception as e:
        logger.error(f"LLM verification failed: {e}")
        return False, "low", f"LLM verification failed: {str(e)}"


def verify_report(
    report_path: str,
    data_dir: str,
    use_llm: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Verify all citations in a report."""
    
    # Load environment
    load_dotenv()
    
    # Read report
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    # Parse citations section
    citation_info = parse_citations_section(report_text)
    logger.info(f"Found {len(citation_info)} citations in Citations section")
    
    # Extract claims with citations
    claims = extract_cited_claims(report_text)
    logger.info(f"Found {len(claims)} sentences with citations")
    
    # Load chunks from preprocessed database
    db_path = os.path.join(data_dir, "long_context.json.chunks.db")
    chunks, all_titles = load_chunks_from_db(db_path)
    logger.info(f"Loaded {len(chunks)} chunks from database ({len(all_titles)} unique titles)")
    
    # Initialize LLM client if needed
    client = None
    if use_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if api_key:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            logger.warning("OPENAI_API_KEY not set, falling back to simple verification")
            use_llm = False
    
    # Verify each citation
    results = []
    sources_not_found = []
    
    for claim in claims:
        for citation in claim.citations:
            source_content, source_found = find_source_content(
                citation, citation_info, chunks, all_titles, data_dir
            )
            
            if not source_found:
                sources_not_found.append({
                    "citation_id": citation.citation_id,
                    "title": citation_info.get(citation.citation_id, Citation("", "")).title,
                    "claim": claim.text[:100]
                })
                results.append(VerificationResult(
                    claim_text=claim.text[:200],
                    citation_id=citation.citation_id,
                    source_content=None,
                    is_supported=False,
                    confidence="low",
                    explanation="Source content not found",
                    source_found=False
                ))
                continue
            
            if citation.citation_type == "Doc":
                # For Doc citations, just check file existence
                results.append(VerificationResult(
                    claim_text=claim.text[:200],
                    citation_id=citation.citation_id,
                    source_content=source_content,
                    is_supported=True,
                    confidence="medium",
                    explanation="File exists (content verification requires file parsing)",
                    source_found=True
                ))
                continue
            
            # Use LLM to verify RAG citations
            if use_llm and client:
                logger.info(f"Verifying {citation.citation_id} with LLM...")
                is_supported, confidence, explanation = verify_with_llm(
                    claim.text, source_content, client
                )
            else:
                # Simple word overlap check as fallback
                claim_words = set(re.findall(r'\b\w{4,}\b', claim.text.lower()))
                source_words = set(re.findall(r'\b\w{4,}\b', source_content.lower()))
                overlap = len(claim_words & source_words) / len(claim_words) if claim_words else 0
                is_supported = overlap >= 0.3
                confidence = "high" if overlap >= 0.6 else "medium" if overlap >= 0.4 else "low"
                explanation = f"Word overlap: {overlap:.0%}"
            
            results.append(VerificationResult(
                claim_text=claim.text[:200],
                citation_id=citation.citation_id,
                source_content=source_content[:500] + "..." if len(source_content) > 500 else source_content,
                is_supported=is_supported,
                confidence=confidence,
                explanation=explanation,
                source_found=True
            ))
    
    # Calculate metrics
    total = len(results)
    supported = sum(1 for r in results if r.is_supported)
    not_supported = total - supported
    sources_found = sum(1 for r in results if r.source_found)
    
    # Calculate scores
    support_rate = supported / total if total > 0 else 0
    source_coverage = sources_found / total if total > 0 else 0
    
    # Weighted score: penalize missing sources more heavily
    citation_quality_score = (support_rate * 0.7 + source_coverage * 0.3) * 100
    
    return {
        "report_path": report_path,
        "data_dir": data_dir,
        "total_citations": total,
        "supported": supported,
        "not_supported": not_supported,
        "sources_found": sources_found,
        "sources_not_found": len(sources_not_found),
        "sources_not_found_details": sources_not_found,
        "metrics": {
            "support_rate": support_rate,
            "source_coverage": source_coverage,
            "citation_quality_score": citation_quality_score
        },
        "results": results
    }


def save_log(verification: Dict[str, Any], log_dir: str):
    """Save verification results to log file."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    report_name = Path(verification["report_path"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{report_name}_{timestamp}.log")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CITATION VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Report: {verification['report_path']}\n")
        f.write(f"Data:   {verification['data_dir']}\n")
        f.write(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("METRICS:\n")
        f.write(f"  Total Citations:      {verification['total_citations']}\n")
        f.write(f"  ✓ Supported:          {verification['supported']}\n")
        f.write(f"  ✗ Not Supported:      {verification['not_supported']}\n")
        f.write(f"  Sources Found:        {verification['sources_found']}\n")
        f.write(f"  Sources Not Found:    {verification['sources_not_found']}\n")
        f.write("\n")
        f.write(f"  Support Rate:         {verification['metrics']['support_rate']:.1%}\n")
        f.write(f"  Source Coverage:      {verification['metrics']['source_coverage']:.1%}\n")
        f.write(f"  Citation Quality:     {verification['metrics']['citation_quality_score']:.1f}/100\n")
        f.write("\n" + "=" * 70 + "\n\n")
        
        if verification['sources_not_found_details']:
            f.write("SOURCES NOT FOUND:\n")
            for item in verification['sources_not_found_details']:
                f.write(f"  - {item['citation_id']}: {item['title']}\n")
            f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("DETAILED RESULTS:\n\n")
        for i, r in enumerate(verification['results'], 1):
            status = "✓" if r.is_supported else "✗"
            found = "Found" if r.source_found else "NOT FOUND"
            f.write(f"[{i}] {status} {r.citation_id} ({r.confidence}) - Source: {found}\n")
            f.write(f"    Claim: {r.claim_text[:150]}...\n")
            f.write(f"    Result: {r.explanation}\n")
            if r.source_content:
                f.write(f"    Source: {r.source_content[:200]}...\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Log saved to: {log_path}")
    return log_path


def print_verification_report(verification: Dict[str, Any], verbose: bool = False):
    """Print verification results."""
    print("\n" + "=" * 70)
    print("CITATION VERIFICATION REPORT")
    print("=" * 70)
    print(f"Report: {verification['report_path']}")
    print(f"Data:   {verification['data_dir']}")
    print("-" * 70)
    print(f"Total Citations:      {verification['total_citations']}")
    print(f"  ✓ Supported:        {verification['supported']}")
    print(f"  ✗ Not Supported:    {verification['not_supported']}")
    print(f"  Sources Found:      {verification['sources_found']}")
    print(f"  Sources Not Found:  {verification['sources_not_found']}")
    print("-" * 70)
    print("METRICS:")
    print(f"  Support Rate:       {verification['metrics']['support_rate']:.1%}")
    print(f"  Source Coverage:    {verification['metrics']['source_coverage']:.1%}")
    print(f"  Citation Quality:   {verification['metrics']['citation_quality_score']:.1f}/100")
    print("=" * 70)
    
    if verification['sources_not_found_details']:
        print("\nSOURCES NOT FOUND:")
        for item in verification['sources_not_found_details']:
            print(f"  - {item['citation_id']}: {item['title']}")
    
    for i, r in enumerate(verification['results'], 1):
        status = "✓" if r.is_supported else "✗"
        found = "" if r.source_found else " [SOURCE NOT FOUND]"
        print(f"\n[{i}] {status} {r.citation_id} ({r.confidence}){found}")
        print(f"    Claim: {r.claim_text[:100]}...")
        print(f"    Result: {r.explanation}")
        if verbose and r.source_content:
            print(f"    Source: {r.source_content[:150]}...")
    
    print("\n" + "=" * 70)


def verify_batch(
    results_dir: str,
    data_dir: str,
    log_dir: str = "eval_log",
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Verify all reports in a results directory.
    
    Args:
        results_dir: Directory containing report markdown files (e.g., results/bench_case1104)
        data_dir: Directory containing data folders (e.g., data/bench_case1104)
        log_dir: Directory for log files
        use_llm: Whether to use LLM for verification
    
    Returns:
        Dictionary with individual results and aggregate metrics
    """
    results_path = Path(results_dir)
    data_path = Path(data_dir)
    
    # Find all report files (support all .md files)
    # Extract case ID from filename patterns like:
    # - 001.md -> case_id = "001"
    # - 001_2025-11-25_16-58-24.md -> case_id = "001"
    report_files = []
    for f in results_path.glob("*.md"):
        # Match patterns: 001.md or 001_timestamp.md
        match = re.match(r'^(\d{3})(?:_.*)?\.md$', f.name)
        if match:
            report_files.append((f, match.group(1)))  # (file_path, case_id)
    
    report_files = sorted(report_files, key=lambda x: x[1])  # Sort by case_id
    logger.info(f"Found {len(report_files)} reports to verify")
    
    # Verify each report
    all_results = []
    individual_metrics = []
    
    for report_file, case_id in report_files:
        case_data_dir = data_path / case_id
        
        if not case_data_dir.exists():
            logger.warning(f"Data directory not found for {case_id}: {case_data_dir}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying: {report_file.name}")
        logger.info(f"Data dir: {case_data_dir}")
        logger.info(f"{'='*60}")
        
        try:
            verification = verify_report(
                str(report_file),
                str(case_data_dir),
                use_llm=use_llm
            )
            
            # Save individual log
            log_path = save_log(verification, log_dir)
            
            # Collect metrics
            individual_metrics.append({
                "case_id": case_id,
                "report_path": str(report_file),
                "total_citations": verification["total_citations"],
                "supported": verification["supported"],
                "not_supported": verification["not_supported"],
                "sources_found": verification["sources_found"],
                "sources_not_found": verification["sources_not_found"],
                "support_rate": verification["metrics"]["support_rate"],
                "source_coverage": verification["metrics"]["source_coverage"],
                "citation_quality_score": verification["metrics"]["citation_quality_score"],
                "log_path": log_path
            })
            
            all_results.append(verification)
            
        except Exception as e:
            logger.error(f"Failed to verify {report_file}: {e}")
            individual_metrics.append({
                "case_id": case_id,
                "report_path": str(report_file),
                "error": str(e)
            })
    
    # Calculate aggregate metrics
    valid_results = [m for m in individual_metrics if "error" not in m]
    
    if valid_results:
        avg_support_rate = sum(m["support_rate"] for m in valid_results) / len(valid_results)
        avg_source_coverage = sum(m["source_coverage"] for m in valid_results) / len(valid_results)
        avg_citation_quality = sum(m["citation_quality_score"] for m in valid_results) / len(valid_results)
        total_citations = sum(m["total_citations"] for m in valid_results)
        total_supported = sum(m["supported"] for m in valid_results)
        total_sources_found = sum(m["sources_found"] for m in valid_results)
    else:
        avg_support_rate = 0
        avg_source_coverage = 0
        avg_citation_quality = 0
        total_citations = 0
        total_supported = 0
        total_sources_found = 0
    
    aggregate_metrics = {
        "total_reports": len(report_files),
        "verified_reports": len(valid_results),
        "failed_reports": len(individual_metrics) - len(valid_results),
        "total_citations": total_citations,
        "total_supported": total_supported,
        "total_sources_found": total_sources_found,
        "overall_support_rate": total_supported / total_citations if total_citations > 0 else 0,
        "overall_source_coverage": total_sources_found / total_citations if total_citations > 0 else 0,
        "avg_support_rate": avg_support_rate,
        "avg_source_coverage": avg_source_coverage,
        "avg_citation_quality_score": avg_citation_quality
    }
    
    batch_result = {
        "results_dir": results_dir,
        "data_dir": data_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_metrics": aggregate_metrics,
        "individual_results": individual_metrics
    }
    
    # Save batch summary
    save_batch_summary(batch_result, log_dir)
    
    return batch_result


def save_batch_summary(batch_result: Dict[str, Any], log_dir: str):
    """Save batch verification summary to files."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON summary
    json_path = os.path.join(log_dir, f"batch_summary_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Batch JSON summary saved to: {json_path}")
    
    # Save human-readable summary
    txt_path = os.path.join(log_dir, f"batch_summary_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH CITATION VERIFICATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Results Directory: {batch_result['results_dir']}\n")
        f.write(f"Data Directory:    {batch_result['data_dir']}\n")
        f.write(f"Timestamp:         {batch_result['timestamp']}\n")
        f.write("-" * 70 + "\n\n")
        
        agg = batch_result['aggregate_metrics']
        f.write("AGGREGATE METRICS:\n")
        f.write(f"  Total Reports:           {agg['total_reports']}\n")
        f.write(f"  Verified Reports:        {agg['verified_reports']}\n")
        f.write(f"  Failed Reports:          {agg['failed_reports']}\n")
        f.write(f"  Total Citations:         {agg['total_citations']}\n")
        f.write(f"  Total Supported:         {agg['total_supported']}\n")
        f.write(f"  Total Sources Found:     {agg['total_sources_found']}\n")
        f.write("\n")
        f.write(f"  Overall Support Rate:    {agg['overall_support_rate']:.1%}\n")
        f.write(f"  Overall Source Coverage: {agg['overall_source_coverage']:.1%}\n")
        f.write(f"  Avg Support Rate:        {agg['avg_support_rate']:.1%}\n")
        f.write(f"  Avg Source Coverage:     {agg['avg_source_coverage']:.1%}\n")
        f.write(f"  Avg Citation Quality:    {agg['avg_citation_quality_score']:.1f}/100\n")
        f.write("\n" + "=" * 70 + "\n\n")
        
        f.write("INDIVIDUAL RESULTS:\n\n")
        f.write(f"{'Case':<8} {'Citations':<12} {'Supported':<12} {'Support%':<12} {'Quality':<12}\n")
        f.write("-" * 60 + "\n")
        
        for result in batch_result['individual_results']:
            if "error" in result:
                f.write(f"{result['case_id']:<8} ERROR: {result['error'][:40]}\n")
            else:
                f.write(f"{result['case_id']:<8} "
                       f"{result['total_citations']:<12} "
                       f"{result['supported']:<12} "
                       f"{result['support_rate']*100:>6.1f}%     "
                       f"{result['citation_quality_score']:>6.1f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    logger.info(f"Batch text summary saved to: {txt_path}")
    
    return json_path, txt_path


def print_batch_summary(batch_result: Dict[str, Any]):
    """Print batch verification summary."""
    print("\n" + "=" * 70)
    print("BATCH CITATION VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Results Directory: {batch_result['results_dir']}")
    print(f"Data Directory:    {batch_result['data_dir']}")
    print(f"Timestamp:         {batch_result['timestamp']}")
    print("-" * 70)
    
    agg = batch_result['aggregate_metrics']
    print("\nAGGREGATE METRICS:")
    print(f"  Total Reports:           {agg['total_reports']}")
    print(f"  Verified Reports:        {agg['verified_reports']}")
    print(f"  Failed Reports:          {agg['failed_reports']}")
    print(f"  Total Citations:         {agg['total_citations']}")
    print(f"  Total Supported:         {agg['total_supported']}")
    print(f"  Total Sources Found:     {agg['total_sources_found']}")
    print()
    print(f"  Overall Support Rate:    {agg['overall_support_rate']:.1%}")
    print(f"  Overall Source Coverage: {agg['overall_source_coverage']:.1%}")
    print(f"  Avg Support Rate:        {agg['avg_support_rate']:.1%}")
    print(f"  Avg Source Coverage:     {agg['avg_source_coverage']:.1%}")
    print(f"  Avg Citation Quality:    {agg['avg_citation_quality_score']:.1f}/100")
    print("\n" + "=" * 70)
    
    print("\nINDIVIDUAL RESULTS:")
    print(f"{'Case':<8} {'Citations':<12} {'Supported':<12} {'Support%':<12} {'Quality':<12}")
    print("-" * 60)
    
    for result in batch_result['individual_results']:
        if "error" in result:
            print(f"{result['case_id']:<8} ERROR: {result['error'][:40]}")
        else:
            print(f"{result['case_id']:<8} "
                  f"{result['total_citations']:<12} "
                  f"{result['supported']:<12} "
                  f"{result['support_rate']*100:>6.1f}%     "
                  f"{result['citation_quality_score']:>6.1f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify citations in reports")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Single report verification
    single_parser = subparsers.add_parser("single", help="Verify a single report")
    single_parser.add_argument("--report", "-r", required=True, help="Path to report markdown file")
    single_parser.add_argument("--data", "-d", required=True, help="Path to data directory")
    single_parser.add_argument("--verbose", "-v", action="store_true", help="Show source content")
    single_parser.add_argument("--no-llm", action="store_true", help="Disable LLM verification")
    single_parser.add_argument("--json", action="store_true", help="Output as JSON")
    single_parser.add_argument("--log-dir", default="eval_log", help="Directory for log files")
    
    # Batch verification
    batch_parser = subparsers.add_parser("batch", help="Verify all reports in a directory")
    batch_parser.add_argument("--results", "-r", required=True, help="Path to results directory")
    batch_parser.add_argument("--data", "-d", required=True, help="Path to data directory")
    batch_parser.add_argument("--log-dir", default="eval_log", help="Directory for log files")
    batch_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.command == "batch":
        # Batch verification
        if not os.path.exists(args.results):
            logger.error(f"Results directory not found: {args.results}")
            sys.exit(1)
        if not os.path.exists(args.data):
            logger.error(f"Data directory not found: {args.data}")
            sys.exit(1)
        
        batch_result = verify_batch(args.results, args.data, args.log_dir, use_llm=True)
        
        if args.json:
            print(json.dumps(batch_result, ensure_ascii=False, indent=2))
        else:
            print_batch_summary(batch_result)
    
    elif args.command == "single":
        # Single report verification
        if not os.path.exists(args.report):
            logger.error(f"Report not found: {args.report}")
            sys.exit(1)
        if not os.path.exists(args.data):
            logger.error(f"Data directory not found: {args.data}")
            sys.exit(1)
        
        verification = verify_report(args.report, args.data, use_llm=not args.no_llm, verbose=args.verbose)
        
        # Save log
        log_path = save_log(verification, args.log_dir)
        
        if args.json:
            output = {
                "report_path": verification["report_path"],
                "total_citations": verification["total_citations"],
                "supported": verification["supported"],
                "not_supported": verification["not_supported"],
                "sources_found": verification["sources_found"],
                "sources_not_found": verification["sources_not_found"],
                "metrics": verification["metrics"],
                "log_path": log_path,
                "results": [
                    {
                        "claim": r.claim_text,
                        "citation_id": r.citation_id,
                        "is_supported": r.is_supported,
                        "confidence": r.confidence,
                        "explanation": r.explanation,
                        "source_found": r.source_found
                    }
                    for r in verification["results"]
                ]
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print_verification_report(verification, args.verbose)
            print(f"\nLog saved to: {log_path}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
