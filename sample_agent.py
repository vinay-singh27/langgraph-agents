import os

from config import OPENAI_API_KEY, TAVILY_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

# fetch the prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# set the llm
llm = ChatOpenAI(model="gpt-3.5-turbo", steaming=True)

agent_runnable = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)



