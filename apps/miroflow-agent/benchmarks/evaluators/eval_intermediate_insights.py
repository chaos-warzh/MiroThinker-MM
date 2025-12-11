#!/usr/bin/env python3
"""
Evaluate intermediate process insights coverage.

This script analyzes execution logs to determine which golden insights
were retrieved during the intermediate steps (tool calls, RAG searches, etc.)
before the final report was generated.

Usage:
    python eval_intermediate_insights.py \
        --execution-log results/results_32k/claude35sonnet1209/002_20251209_145220/execution_log.json \
        --golden-insights golden_insights/002.json \
        --output evaluations/002_evaluation.json
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class InsightMatch:
    """Represents a match between a golden insight and content in the log."""
    insight_index: int
    insight_text: str
    matched_in_turn: int
    matched_role: str  # 'user' or 'assistant'
    match_type: str  # 'exact', 'partial', 'keyword'
    matched_content_snippet: str
    confidence: float  # 0.0 to 1.0


@dataclass
class EvaluationResult:
    """Overall evaluation result."""
    total_insights: int
    insights_found_in_intermediate: int
    insights_found_in_final: int
    insights_not_found: int
    coverage_rate: float
    intermediate_coverage_rate: float
    insights_details: List[Dict]


def load_execution_log(log_path: str) -> Dict:
    """Load and parse the execution log JSON file."""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_golden_insights(insights_path: str) -> List[str]:
    """Load golden insights from JSON file."""
    with open(insights_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and 'gold_insights' in data:
        return [item['insight'] for item in data['gold_insights']]
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return data
        elif all(isinstance(item, dict) and 'insight' in item for item in data):
            return [item['insight'] for item in data]
    
    raise ValueError(f"Unknown golden insights format in {insights_path}")


def extract_content_from_message(message: Dict) -> str:
    """Extract text content from a message, handling different content formats."""
    content = message.get('content', '')
    
    # Handle string content
    if isinstance(content, str):
        return content
    
    # Handle list content (e.g., multimodal messages with text and images)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Handle text blocks
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                # Handle tool results
                elif item.get('type') == 'tool_result':
                    text_parts.append(str(item.get('content', '')))
        return '\n'.join(text_parts)
    
    return str(content)


def extract_conversations_from_log(log_data: Dict) -> List[Dict]:
    """
    Extract conversation turns from execution log.
    
    Supports multiple log formats:
    1. New format: main_agent_message_history.message_history
    2. Legacy format: step_logs or messages array
    
    Returns a list of dicts with:
    - turn: turn number
    - role: 'user' or 'assistant'
    - content: the message content
    - source: where the content came from
    """
    conversations = []
    
    # Try new format first: main_agent_message_history
    if 'main_agent_message_history' in log_data:
        agent_history = log_data['main_agent_message_history']
        message_history = agent_history.get('message_history', [])
        
        for i, msg in enumerate(message_history):
            role = msg.get('role', '')
            content = extract_content_from_message(msg)
            
            if role and content:
                # Calculate turn number (user-assistant pairs)
                turn = i // 2
                conversations.append({
                    'turn': turn,
                    'role': role,
                    'content': content,
                    'source': 'main_agent_message_history',
                    'message_index': i
                })
        
        # If we found conversations in the new format, return them
        if conversations:
            return conversations
    
    # Try legacy format: step_logs
    step_logs = log_data.get('step_logs', [])
    if step_logs:
        current_turn = 0
        for step in step_logs:
            step_name = step.get('step_name', '')
            message = step.get('message', '')
            metadata = step.get('metadata', {})
            
            # Detect turn changes
            if "Turn:" in step_name:
                turn_match = re.search(r'Turn:\s*(\d+)', step_name)
                if turn_match:
                    current_turn = int(turn_match.group(1))
            
            # Extract tool call results (these are "user" responses in the conversation)
            if "Tool Call Success" in step_name:
                tool_result = metadata.get('result', '')
                if tool_result:
                    conversations.append({
                        'turn': current_turn,
                        'role': 'user',
                        'content': str(tool_result),
                        'source': 'tool_result'
                    })
            
            # Also check for RAG results
            if "rag" in step_name.lower() or "rag" in message.lower():
                if metadata.get('result'):
                    conversations.append({
                        'turn': current_turn,
                        'role': 'user',
                        'content': str(metadata.get('result', '')),
                        'source': 'rag_result'
                    })
    
    # Try legacy format: messages array
    messages = log_data.get('messages', [])
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = extract_content_from_message(msg)
            if role and content:
                conversations.append({
                    'turn': i // 2,  # Approximate turn number
                    'role': role,
                    'content': content,
                    'source': 'messages_array'
                })
    
    return conversations


def extract_key_phrases(insight: str) -> List[str]:
    """Extract key phrases from an insight for matching."""
    # Remove common words and extract meaningful phrases
    phrases = []
    
    # Extract quoted content
    quoted = re.findall(r'["""]([^"""]+)["""]', insight)
    phrases.extend(quoted)
    
    # Extract numbers and percentages
    numbers = re.findall(r'\d+(?:\.\d+)?(?:%|万|亿|元)?', insight)
    phrases.extend(numbers)
    
    # Extract proper nouns (simplified: consecutive Chinese characters or capitalized words)
    proper_nouns = re.findall(r'[A-Z][a-zA-Z]+|[\u4e00-\u9fa5]{2,8}(?:公司|平台|法院|案|法|条例|规定|行动)', insight)
    phrases.extend(proper_nouns)
    
    # Extract key terms related to live streaming and e-commerce
    key_terms = [
        '避风港', '红旗原则', '通知删除', '连带责任', '虚假宣传', '欺诈',
        '直播带货', '主播', '平台', 'GMV', '罚款', '赔偿', '侵权',
        '未成年人', '打赏', '退款', '算法', '大数据杀熟', 'TikTok',
        '跨境电商', '知识产权', '信息网络传播权', '清朗', '专项行动',
        '市场监管', '消费者权益', '七日无理由退货', '假一赔十',
        '刷单炒信', '虚假评价', '商标侵权', '音著协', '斗鱼',
        '抖音', '海淀区法院', '最高法', '典型案例'
    ]
    for term in key_terms:
        if term in insight:
            phrases.append(term)
    
    return list(set(phrases))


def calculate_match_score(insight: str, content: str) -> Tuple[float, str]:
    """
    Calculate how well the content matches the insight.
    
    Returns:
    - score: 0.0 to 1.0
    - match_type: 'exact', 'partial', 'keyword', 'none'
    """
    insight_lower = insight.lower()
    content_lower = content.lower()
    
    # Check for exact match (unlikely but possible)
    if insight in content:
        return 1.0, 'exact'
    
    # Extract key phrases and check coverage
    key_phrases = extract_key_phrases(insight)
    if not key_phrases:
        # Fallback to simple substring matching
        words = [w for w in insight.split() if len(w) > 2]
        key_phrases = words[:10]  # Take first 10 meaningful words
    
    matched_phrases = []
    for phrase in key_phrases:
        if phrase.lower() in content_lower:
            matched_phrases.append(phrase)
    
    if not key_phrases:
        return 0.0, 'none'
    
    coverage = len(matched_phrases) / len(key_phrases)
    
    if coverage >= 0.7:
        return coverage, 'partial'
    elif coverage >= 0.3:
        return coverage, 'keyword'
    else:
        return coverage, 'none'


def find_insight_in_conversations(
    insight: str,
    insight_index: int,
    conversations: List[Dict],
    exclude_final: bool = True
) -> List[InsightMatch]:
    """
    Find where an insight appears in the conversation history.
    
    Args:
        insight: The golden insight text
        insight_index: Index of the insight
        conversations: List of conversation turns
        exclude_final: If True, exclude the final report generation turn
    
    Returns:
        List of InsightMatch objects
    """
    matches = []
    
    if not conversations:
        return matches
    
    # Determine the final turn (usually the last few turns are report generation)
    max_turn = max((c['turn'] for c in conversations), default=0)
    
    for conv in conversations:
        # Optionally exclude final turns (report generation)
        if exclude_final and conv['turn'] >= max_turn - 1:
            continue
        
        content = conv['content']
        score, match_type = calculate_match_score(insight, content)
        
        if score >= 0.3:  # Threshold for considering it a match
            # Extract a snippet around the match
            snippet = content[:500] + "..." if len(content) > 500 else content
            
            matches.append(InsightMatch(
                insight_index=insight_index,
                insight_text=insight[:200] + "..." if len(insight) > 200 else insight,
                matched_in_turn=conv['turn'],
                matched_role=conv['role'],
                match_type=match_type,
                matched_content_snippet=snippet,
                confidence=score
            ))
    
    return matches


def evaluate_insights_coverage(
    execution_log_path: str,
    golden_insights_path: str,
    final_report_path: str = None
) -> EvaluationResult:
    """
    Main evaluation function.
    
    Args:
        execution_log_path: Path to execution_log.json
        golden_insights_path: Path to golden insights JSON
        final_report_path: Optional path to final_report.md
    
    Returns:
        EvaluationResult with detailed analysis
    """
    # Load data
    log_data = load_execution_log(execution_log_path)
    insights = load_golden_insights(golden_insights_path)
    
    # Extract conversations
    conversations = extract_conversations_from_log(log_data)
    
    print(f"📊 Extracted {len(conversations)} conversation turns from log")
    
    # Load final report if available
    final_report = ""
    if final_report_path and os.path.exists(final_report_path):
        with open(final_report_path, 'r', encoding='utf-8') as f:
            final_report = f.read()
    else:
        # Try to find final report in the same directory
        log_dir = os.path.dirname(execution_log_path)
        possible_report = os.path.join(log_dir, 'final_report.md')
        if os.path.exists(possible_report):
            with open(possible_report, 'r', encoding='utf-8') as f:
                final_report = f.read()
    
    # Evaluate each insight
    insights_details = []
    insights_found_intermediate = 0
    insights_found_final = 0
    
    for i, insight in enumerate(insights):
        # Find in intermediate steps
        intermediate_matches = find_insight_in_conversations(
            insight, i, conversations, exclude_final=True
        )
        
        # Check if in final report
        final_score, final_match_type = calculate_match_score(insight, final_report)
        in_final = final_score >= 0.3
        
        # Determine best intermediate match
        best_intermediate_match = None
        if intermediate_matches:
            best_intermediate_match = max(intermediate_matches, key=lambda m: m.confidence)
            insights_found_intermediate += 1
        
        if in_final:
            insights_found_final += 1
        
        # Determine status
        if intermediate_matches and in_final:
            status = 'found_both'
        elif intermediate_matches:
            status = 'found_intermediate_only'
        elif in_final:
            status = 'found_final_only'
        else:
            status = 'not_found'
        
        detail = {
            'insight_index': i,
            'insight_text': insight,
            'found_in_intermediate': len(intermediate_matches) > 0,
            'intermediate_matches_count': len(intermediate_matches),
            'best_intermediate_confidence': best_intermediate_match.confidence if best_intermediate_match else 0,
            'best_intermediate_turn': best_intermediate_match.matched_in_turn if best_intermediate_match else None,
            'found_in_final': in_final,
            'final_confidence': final_score,
            'status': status
        }
        insights_details.append(detail)
    
    # Calculate overall metrics
    total = len(insights)
    not_found = total - max(insights_found_intermediate, insights_found_final)
    
    intermediate_rate = insights_found_intermediate / total if total > 0 else 0
    final_rate = insights_found_final / total if total > 0 else 0
    
    return EvaluationResult(
        total_insights=total,
        insights_found_in_intermediate=insights_found_intermediate,
        insights_found_in_final=insights_found_final,
        insights_not_found=not_found,
        coverage_rate=final_rate,
        intermediate_coverage_rate=intermediate_rate,
        insights_details=insights_details
    )


def print_evaluation_report(result: EvaluationResult):
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("INTERMEDIATE INSIGHTS EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Golden Insights: {result.total_insights}")
    print(f"   Found in Intermediate Steps: {result.insights_found_in_intermediate} ({result.intermediate_coverage_rate:.1%})")
    print(f"   Found in Final Report: {result.insights_found_in_final} ({result.coverage_rate:.1%})")
    print(f"   Not Found: {result.insights_not_found}")
    
    print(f"\n📋 Detailed Breakdown:")
    print("-" * 70)
    
    for detail in result.insights_details:
        status_emoji = {
            'found_both': '✅',
            'found_intermediate_only': '🔶',
            'found_final_only': '📝',
            'not_found': '❌'
        }.get(detail['status'], '❓')
        
        print(f"\n{status_emoji} Insight #{detail['insight_index'] + 1}:")
        insight_preview = detail['insight_text'][:100] + "..." if len(detail['insight_text']) > 100 else detail['insight_text']
        print(f"   Text: {insight_preview}")
        print(f"   Status: {detail['status']}")
        if detail['found_in_intermediate']:
            print(f"   Intermediate: Turn {detail['best_intermediate_turn']}, Confidence: {detail['best_intermediate_confidence']:.2f}")
        if detail['found_in_final']:
            print(f"   Final Report: Confidence: {detail['final_confidence']:.2f}")
    
    print("\n" + "=" * 70)


def get_default_output_dir() -> Path:
    """Get the default evaluations output directory."""
    # Create evaluations folder at the same level as results
    script_dir = Path(__file__).parent.parent.parent  # Go up to miroflow-agent
    evaluations_dir = script_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    return evaluations_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate intermediate process insights coverage"
    )
    parser.add_argument(
        "--execution-log", "-e",
        required=True,
        help="Path to execution_log.json"
    )
    parser.add_argument(
        "--golden-insights", "-g",
        required=True,
        help="Path to golden insights JSON file"
    )
    parser.add_argument(
        "--final-report", "-f",
        default=None,
        help="Path to final_report.md (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save evaluation result JSON (optional, defaults to evaluations/ folder)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    result = evaluate_insights_coverage(
        execution_log_path=args.execution_log,
        golden_insights_path=args.golden_insights,
        final_report_path=args.final_report
    )
    
    # Print report
    print_evaluation_report(result)
    
    # Determine output path
    output_path = args.output
    if not output_path:
        # Generate default output path in evaluations folder
        evaluations_dir = get_default_output_dir()
        
        # Extract task ID from execution log path
        log_path = Path(args.execution_log)
        task_id = log_path.parent.name.split('_')[0]  # e.g., "002" from "002_20251209_145220"
        model_name = log_path.parent.parent.name  # e.g., "claude35sonnet1209"
        context_size = log_path.parent.parent.parent.name  # e.g., "results_32k"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = evaluations_dir / f"{context_size}_{model_name}_{task_id}_{timestamp}.json"
    
    # Save to file
    output_data = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'execution_log_path': str(args.execution_log),
        'golden_insights_path': str(args.golden_insights),
        'total_insights': result.total_insights,
        'insights_found_in_intermediate': result.insights_found_in_intermediate,
        'insights_found_in_final': result.insights_found_in_final,
        'insights_not_found': result.insights_not_found,
        'coverage_rate': result.coverage_rate,
        'intermediate_coverage_rate': result.intermediate_coverage_rate,
        'insights_details': result.insights_details
    }
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
