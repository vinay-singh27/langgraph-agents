from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a hiring manager at a top-tier company, evaluating a candidate's resume. Provide a detailed "
            "critique and actionable recommendations for improvement, focusing on clarity, impact, relevance, "
            "and style. Highlight areas to improve such as word choice, quantifiable achievements, formatting, "
            "and alignment with the target role. Ensure the suggestions enhance the resume's effectiveness in "
            "showcasing the candidate's skills and experience."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional resume assistant specializing in crafting impactful resumes for high-level roles. "
            "Generate the best possible resume based on the user's instructions, tailoring it to the target role and "
            "industry. If the user provides feedback or critique, revise your previous version to address their "
            "concerns and improve the content. Focus on presenting achievements, skills, and experiences in a "
            "compelling and concise manner, ensuring alignment with the user's goals and industry standards."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(model="gpt-4o")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm




