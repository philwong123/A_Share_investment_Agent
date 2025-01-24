from datetime import datetime, timedelta
import argparse
from agents.valuation import valuation_agent
from agents.state import AgentState
from agents.sentiment import sentiment_agent
from agents.risk_manager import risk_management_agent
from agents.technicals import technical_analyst_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.market_data import market_data_agent
from agents.fundamentals import fundamentals_agent
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import akshare as ak
import pandas as pd
import json


def format_trading_result(result_str: str) -> None:
    """美化并翻译交易结果"""
    try:
        print("开始处理交易结果...")
        # 更完整的字符串清理
        cleaned_str = result_str.strip('"')  # 移除外层的引号
        if cleaned_str.startswith('```'):
            print("检测到Markdown代码块，正在清理...")
            # 移除```json标记和结尾的```
            cleaned_str = cleaned_str.replace('```json', '')
            cleaned_str = cleaned_str.replace('```', '')
            cleaned_str = cleaned_str.strip()
        
        # 处理转义字符
        cleaned_str = cleaned_str.replace('\\n', '\n')
        cleaned_str = cleaned_str.replace('\\"', '"')
        
        print(f"清理后的JSON字符串:\n{cleaned_str}")
        
        # 解析JSON字符串
        result = json.loads(cleaned_str)
        print("JSON解析成功")
        
        # 交易动作翻译映射
        action_map = {
            "buy": "买入",
            "sell": "卖出",
            "hold": "持有"
        }
        
        # 信号翻译映射
        signal_map = {
            "bullish": "看涨",
            "bearish": "看跌",
            "neutral": "中性"
        }
        
        # 代理名称翻译映射
        agent_map = {
            "Technical Analysis": "技术分析",
            "Fundamental Analysis": "基本面分析",
            "Sentiment Analysis": "情绪分析",
            "Valuation Analysis": "估值分析",
            "Risk Management": "风险管理"
        }
        
        print("\n========== 交易决策结果 ==========")
        print(f"交易动作: {action_map.get(result['action'], result['action'])}")
        print(f"交易数量: {result['quantity']}")
        print(f"决策置信度: {result['confidence']*100:.0f}%")
        
        print("\n各分析师意见:")
        for signal in result['agent_signals']:
            confidence = f"{signal['confidence']*100:.0f}%" if signal['confidence'] is not None else "N/A"
            print(f"- {agent_map.get(signal['agent_name'], signal['agent_name'])}: "
                  f"{signal_map.get(signal['signal'], signal['signal'])} "
                  f"(置信度: {confidence})")
        
        print("\n决策理由:")
        # 使用智谱AI翻译决策理由
        reasoning_en = result['reasoning']
        reasoning_zh = translate_to_chinese(reasoning_en)  # 添加翻译功能
        print(reasoning_zh)
        print("================================")
        
    except json.JSONDecodeError as e:
        print(f"结果格式化失败，JSON解析错误: {str(e)}")
        print("清理后的字符串：")
        print(cleaned_str)
        print("\n原始结果：")
        print(result_str)
    except Exception as e:
        print(f"结果处理时发生错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        print(f"错误位置: {e.__traceback__.tb_lineno}")
        print("原始结果：")
        print(result_str)

def translate_to_chinese(text: str) -> str:
    """使用智谱AI翻译英文文本到中文"""
    try:
        print("\n开始翻译文本...")
        from tools.openrouter_config import ai_service
        
        prompt = f"""
        请将以下英文文本翻译成中文，保持专业性和准确性：
        
        {text}
        
        只需要返回翻译结果，不要包含原文或其他解释。
        """
        
        print("调用AI服务...")
        response = ai_service.generate_content(prompt.strip())
        print(f"AI服务响应类型: {type(response)}")
        print(f"AI服务响应内容: {response}")
        
        if not response:
            print("翻译服务返回空响应")
            return text
            
        # 如果response是字符串，直接返回
        if isinstance(response, str):
            print("响应是字符串类型")
            return response
            
        # 如果response是其他类型的对象，尝试获取text属性
        if hasattr(response, 'text'):
            print("响应对象具有text属性")
            return response.text
            
        # 如果都不是，返回原文
        print(f"无法解析翻译响应: {response}")
        return text
        
    except Exception as e:
        print(f"翻译失败: {str(e)}")
        print(f"错误类型: {type(e)}")
        print(f"错误位置: {e.__traceback__.tb_lineno}")
        return text  # 如果翻译失败，返回原文


##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, num_of_news: int = 5):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": num_of_news,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content


# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)
workflow.add_node("valuation_agent", valuation_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")
workflow.add_edge("technical_analyst_agent", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("sentiment_agent", "risk_management_agent")
workflow.add_edge("valuation_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD). Defaults to 1 year before end date')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD). Defaults to yesterday')
    parser.add_argument('--show-reasoning', action='store_true',
                        help='Show reasoning from each agent')
    parser.add_argument('--num-of-news', type=int, default=5,
                        help='Number of news articles to analyze for sentiment (default: 5)')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial cash amount (default: 100,000)')
    parser.add_argument('--initial-position', type=int, default=0,
                        help='Initial stock position (default: 0)')

    args = parser.parse_args()

    # Set end date to yesterday if not specified
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday if not args.end_date else min(
        datetime.strptime(args.end_date, '%Y-%m-%d'), yesterday)

    # Set start date to one year before end date if not specified
    if not args.start_date:
        start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    # Validate dates
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")

    # Validate num_of_news
    if args.num_of_news < 1:
        raise ValueError("Number of news articles must be at least 1")
    if args.num_of_news > 100:
        raise ValueError("Number of news articles cannot exceed 100")

    # Configure portfolio
    portfolio = {
        "cash": args.initial_capital,
        "stock": args.initial_position
    }

    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news
    )
    
    format_trading_result(result)


def get_historical_data(symbol: str) -> pd.DataFrame:
    """Get historical market data for a given stock symbol.
    If we can't get the full year of data, use whatever is available."""
    # Calculate date range
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday  # Use yesterday as end date
    target_start_date = yesterday - \
        timedelta(days=365)  # Target: 1 year of data

    print(f"\n正在获取 {symbol} 的历史行情数据...")
    print(f"目标开始日期：{target_start_date.strftime('%Y-%m-%d')}")
    print(f"结束日期：{end_date.strftime('%Y-%m-%d')}")

    try:
        # Get historical data
        df = ak.stock_zh_a_hist(symbol=symbol,
                                period="daily",
                                start_date=target_start_date.strftime(
                                    "%Y%m%d"),
                                end_date=end_date.strftime("%Y%m%d"),
                                adjust="qfq")

        actual_days = len(df)
        target_days = 365  # Target: 1 year of data

        if actual_days < target_days:
            print(f"提示：实际获取到的数据天数({actual_days}天)少于目标天数({target_days}天)")
            print(f"将使用可获取到的所有数据进行分析")

        print(f"成功获取历史行情数据，共 {actual_days} 条记录\n")
        return df

    except Exception as e:
        print(f"获取历史数据时发生错误: {str(e)}")
        print("将尝试获取最近可用的数据...")

        # Try to get whatever data is available
        try:
            df = ak.stock_zh_a_hist(symbol=symbol,
                                    period="daily",
                                    adjust="qfq")
            print(f"成功获取历史行情数据，共 {len(df)} 条记录\n")
            return df
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return pd.DataFrame()
