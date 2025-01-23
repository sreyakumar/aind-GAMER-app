from langchain_aws import BedrockLLM
from langsmith import Client
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage
from typing import List

def create_bedrock_chain():
    # Initialize Bedrock LLM
    llm = BedrockLLM(
        model_id="anthropic.claude-v2",
        model_kwargs={
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
        }
    )

    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}")
    ])

    # Create the chain
    chain = prompt | llm

    return chain

def process_with_tracing(user_input: str, project_name: str = "bedrock-test"):
    try:
        # Initialize LangSmith client
        client = Client()

        print(client)
        
        # Create the chain
        chain = create_bedrock_chain()

        # Run with tracing
        with client.trace(
            project_name=project_name,
            tags=["bedrock", "claude-v2"]
        ) as trace:
            response = chain.invoke(
                {"input": user_input}
            )
            return response
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Test input
        user_input = "Hello, world! Please explain what you are."
        
        # Run the pipeline
        response = process_with_tracing(user_input)
        print("\nResponse:", response)
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")