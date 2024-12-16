from typing import List, Sequence

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph
from langgraph_env.chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(state: Sequence[BaseMessage]):
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    inputs = HumanMessage(content="""Improve this resume:
Netomi Gurgaon
Senior Machine Learning Engineer Nov 2023 – present
• Lead R&D initiatives within the Data Science team, focusing on the development of cutting-edge features and optimizations to
enhance product conversation capability.
• Large Action Model: LLM-based Agent with capability to perform user actions
- Developed LangChain Agent Framework using gpt-4o and llama 3.1 LLM models that dynamically generates tools from backend
JSON data using the StructuredTool dataclass. These tools are optimized to effectively provide user travel itineraries and
recommend alternative flights in case of delays, while also facilitating human-like conversations.
- Deployed production-ready code, capable of handling 3 TPS (transactions per second) of traffic, enabling response streaming,
caching tool data, and publishing Kafka topics to share final outputs with the analytics service for real-time insights.
• Fine-Tuned and Deployed Llama 70B Model for Enhanced Response Generation: Utilized Axolotl to fine-tune the Llama 70B
model on RunPod serverless GPU instance. Post-tuning, deployed the model on RunPod’s VLLM serverless platform, ensuring
scalable and efficient real-time responses.
ZS Associate Pune
Senior Data Scientist Oct 2019 – Oct 2023
• Kural MR: Scalable application that leverages GenAI to generate insights from Market Research (MR) studies
- Created a scalable knowledge graph and search engine for Market Research insights, using GPT-3.5 and LangChain to extract
insights from transcripts, resulting in improved data accessibility.
- Containerized the application as distinct microservices, managed with TeamCity and deployed. Deployed the containers on AWS
EKS to manage the Kubernetes deployments and services.
• ZAIDYN OLI: ML-based product to provide comprehensive insights on Opinion Leaders
- Developed a complete ML ecosystem to identify and profile the Opinion Leaders and deployed it in multiple countries.
- Achieved 71% precision and 62% recall in predicting the Opinion Leaders, which is ~4x increase in f1-score
- Worked on creating Network Graph and applied Louvain Algorithm to detect the local communities. Used the PageRank
algorithm to identify local influencers in these communities.
• Market Share Estimation Tool: Determined the addressable undiagnosed patient opportunity
- Utilized the genetic algorithm for feature selection and implemented the Positive-Unlabelled algorithm to achieve 78% recall.
Identified an additional pool of 1M undiagnosed patients as potential opportunities.
- Leveraged PySpark on databricks to handle the transactional data of 50 billion records of the patients
Hero MotoCorp Gurgaon
Data Analyst Jul 2018 - Sep 2019
    """)
    response = graph.invoke(inputs)
