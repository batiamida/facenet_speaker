from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langgraph.checkpoint.memory import MemorySaver
import argparse

from face_recognition.authorize_face import FaceDB
from face_recognition.ai_player import AIPlayer
from speaker_decisions.audio_chat import Transcriber
from typing import TypedDict
import numpy as np

from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile")
model = SentenceTransformer("all-MiniLM-L6-v2")
transcriber = Transcriber()


class State(TypedDict):
    username: str
    user_input: str
    playlists_descr: list[dict]
    response: str
    playlist_name: str
    playlist_id: str
    ready_for_rag: bool
    play_song: bool


class ModelResponse(TypedDict):
    response: str
    ready_for_rag: bool


def model_node(state: State) -> dict:
    playlist_exmpl = " ".join([val for col, val in state["playlists_descr"][0].items() if col in ["playlist_name", "genre", "mood", "description"]])
    sys_prompt_base_llm = f"""
    You are AI assistant that helps customers to find best suited playlist to play.
    You shouldn't mention any information about playlists you see only asking clarifying questions.
    You have rag tool that will be triggered if you find out what kind of playlist user wants to hear.
    It shouldn't be exact match if user is not sure. Your job just to recommend
    You will be given username to interact with user, playlists that you can choose from to recommend for user or ask user any clarifying questions and user_input which is response from user for interaction
    ### Instructions:
    Keep the conversation more short, don't ask too long questions
    When you are ready Return 
    ready_for_rag=True: whenever you are sure that you picked
    and field response exists for interaction with the user and whenever you are ready to describe playlist then write short description that suits the best
    Here is example of description that describes random playlist
    Don't repeat yourself and give only recommendations and nothing else except that don't ask repeatedly questions
    {playlist_exmpl}

    Take into account if in memory user already said to you something that he likes and you can find possible playlist just use rag and retrieve this playlist
    Take into account that if user didn't provide certain information about genre or certain names you can omit those and describe playlist with the information you have
    """
    system_base_message = SystemMessagePromptTemplate.from_template(sys_prompt_base_llm)
    human_base_message = HumanMessagePromptTemplate.from_template("""
                                                                ### Input from user 
                                                                {username}
                                                                {playlists}
                                                                {user_input}
                                                            """)

    prompt = ChatPromptTemplate.from_messages([system_base_message, human_base_message])
    decision_model = prompt | llm.with_structured_output(ModelResponse)
    res = decision_model.invoke({"username": state["username"],
                                 "playlists": state["playlists_descr"],
                                 "user_input": state["user_input"]
                                 })

    state["ready_for_rag"] = res["ready_for_rag"]
    state["response"] = res["response"]
    return state


def rag_node(state: State) -> State:
    query = state["response"]
    query_embedding = model.encode([query])
    playlists_embeddings = model.encode([" ".join((val["playlist_name"], val["playlist_genre"],
                                                  val["playlist_mood"], val["playlist_description"])) for val in state["playlists_descr"]])
    sim = np.dot(playlists_embeddings, query_embedding[0])
    top_idx = np.argmax(sim)

    top_playlist = state["playlists_descr"][top_idx]
    state["playlist_name"] = top_playlist["playlist_name"]
    state["playlist_id"] = top_playlist["playlist_id"]
    state["play_song"] = True

    return state


def tools_node(state: State) -> State:
    playlist_id = state["playlist_id"]
    print("🎶 Playing now...")
    AIPlayer.open_ytmusic_plyalist(playlist_id)
    return state


def get_user_input_node_audio(state: State) -> dict:
    last_response = state.get("response", "what kind of music you want to listen?")
    print(f"\n🤖 {last_response}")
    transcriber.say(last_response)
    user_input = transcriber.record_and_transcribe()
    print("🧑 You: " + user_input)
    state["user_input"] = user_input

    return state


def get_user_input_node_text(state: State) -> dict:
    last_response = state.get("response", "How can I help you?")
    print(f"\n🤖 {last_response}")
    user_input = input("Write something: ")
    print("🧑 You: " + user_input)
    state["user_input"] = user_input

    return state


parser = argparse.ArgumentParser(description="Chat argument parser")
parser.add_argument("--mode", choices=["audio", "text"], help="Output file name", default="text")
args = parser.parse_args()
mode_func = get_user_input_node_audio\
        if args.mode == "audio"\
        else get_user_input_node_text

workflow = StateGraph(State)

workflow.add_node("model", model_node)
workflow.add_node("get_user_input", mode_func)
workflow.add_node("rag", rag_node)
workflow.add_node("play_music", tools_node)

workflow.set_entry_point("get_user_input")

workflow.add_edge("get_user_input", "model")

workflow.add_conditional_edges(
    "model",
    lambda state: "rag" if state.get("ready_for_rag") else "get_user_input"
)

workflow.add_edge("rag", "play_music")
workflow.add_edge("play_music", END)


if __name__ == "__main__":
    fdb = FaceDB(r'your_path_to_db')
    user_id, username = fdb.authorize_user_cam()
    playlists = fdb.get_playlists_by_userid(user_id)
    state = State(**{"username": username, "playlists_descr": playlists})

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    while True:
        res = graph.invoke(
            state,
            config,
        )
        state.update(res)

        if res.get("ready_for_rag"):
            break