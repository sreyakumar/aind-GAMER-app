import asyncio
import os
import time
import uuid
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from metadata_chatbot.agents.async_workflow import async_workflow
from metadata_chatbot.agents.mongo_db_agent import astream_input


warnings.filterwarnings("ignore")

load_dotenv()


@st.cache_resource
def load_checkpointer():
    return MemorySaver()


async def answer_generation(
    query: str, chat_history: list, config: dict, model
):
    inputs = {
        "messages": chat_history,
    }
    async for output in model.astream(inputs, config):
        for key, value in output.items():
            if key != "database_query":
                yield value["messages"][0].content
            else:
                try:
                    query = str(chat_history) + query
                    async for result in astream_input(query=query):
                        response = result["type"]
                        if response == "intermediate_steps":
                            yield result["content"]
                        if response == "agg_pipeline":
                            yield "The MongoDB pipeline used is:"
                            yield f"`{result['content']}`"
                        if response == "tool_response":
                            yield "Retrieved output from MongoDB:"
                            yield f"""```json
                                    {result['content']}
                                    ```"""
                        if response == "final_answer":
                            yield result["content"]
                except Exception as e:
                    yield (
                        "An error has occured with the retrieval from DocDB: "
                        f"{e}. Try structuring your query another way."
                    )


def set_query(query):
    st.session_state.query = query


async def main():
    st.title("GAMER: Generative Analysis of Metadata Retrieval")

    langchain_api_key = os.getenv(
        "LANGCHAIN_API_KEY"
    )  # st.secrets.get("LANGCHAIN_API_KEY")
    langchain_endpoint = os.getenv(
        "LANGCHAIN_ENDPOINT"
    )  # "https://api.smith.langchain.com"
    project = os.getenv("LANGSMITH_PROJECT")
    client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)

    ls_tracer = LangChainTracer(project_name=project, client=client)
    run_collector = RunCollectorCallbackHandler()
    cfg = RunnableConfig()
    cfg["callbacks"] = [ls_tracer, run_collector]

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = ""
    st.session_state.thread_id = str(uuid.uuid4())

    checkpointer = load_checkpointer()
    model = async_workflow.compile(checkpointer=checkpointer)

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "run_id" not in st.session_state:
        st.session_state.run_id= None

    st.info(
        "Ask a question about the AIND metadata! "
        "Please note that it will take a few seconds to generate an answer. "
        "Type a query to start or pick one of these suggestions:"
    )

    examples = [
        (
            "What are the modalities that exist in the database? "
            "What are the least and most common ones?"
        ),
        (
            "What is the MongoDB query to find the injections used in "
            "SmartSPIM_675387_2023-05-23_23-05-56"
        ),
        (
            "Can you list all the procedures performed on 662616, "
            "including their start and end dates?"
        ),
    ]

    columns = st.columns(len(examples))
    for i, column in enumerate(columns):
        with column:
            st.button(examples[i], on_click=set_query, args=[examples[i]])

    message = st.chat_message("assistant")
    message.write("Hello! How can I help you?")

    user_query = st.chat_input("Message GAMER")

    if user_query:
        st.session_state.query = user_query

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    query = st.session_state.query
    if query is not None and query != "":
        st.session_state.messages.append(HumanMessage(query))

        # with st.chat_message("user"):
        #     st.markdown(query)

        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }
            prev = None
            generation = None
            chat_history = st.session_state.messages
            with collect_runs() as cb:
                with st.status(
                    "Generating answer...", expanded=True
                ) as status:
                    async for result in answer_generation(
                        query, chat_history, config, model
                    ):
                        if prev is not None:
                            st.markdown(prev)
                        prev = result
                        generation = prev
                    status.update(label="Answer generation successful.")
                    st.session_state.run_id = cb.traced_runs[-1].id
                    st.session_state.messages.append(AIMessage(generation))
            final_response = st.empty()
            final_response.markdown(generation)
                

    @st.cache_data(ttl="2h", show_spinner=False)
    def get_run_url(run_id):
        time.sleep(1)
        return client.read_run(run_id).url

    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        score_mappings = {
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings["faces"]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"FACES: {feedback['score']}"

                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }

            else:
                st.warning("Invalid feedback score.")

    st.session_state.query = ""


if __name__ == "__main__":
    asyncio.run(main())