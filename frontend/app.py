import streamlit as st
import uuid
from util import generate
from db import save_conversation, save_feedback, init_db
from streamlit import session_state as ss


def print_log(message):
    print(message, flush=True)


def feedback_cb():
    """Processes feedback."""
    ss.fbdata = {"feedback_thumb": ss.thumbs, "feedback_comment": ss.comment}
    scores = {"👍": 1, "👎": -1}
    score = scores.get(ss.thumbs)
    print_log(f"User gave feedback:  {score}")
    print_log(f"User gave feedback:  {ss.comment}")
    save_feedback(st.session_state.conversation_id, score, ss.comment)
    print_log("Feedback saved to database")


def main():
    init_db()
    print_log("Initializing the Assistant")
    st.title("RAG Assistant")

    # Session state initialization
    st.session_state.conversation_id = str(uuid.uuid4())
    print_log(f"New conversation started with ID: {st.session_state.conversation_id}")

    # Model selection
    model_choice = st.selectbox(
        "Select a model:",
        ["llama3.1-8b", "llama3.1-70b"],
    )
    print_log(f"User selected model: {model_choice}")

    # Search selection
    search_type = st.radio("Select search type:", ["Text", "Vector"])
    print_log(f"User selected search type: {search_type}")

    reranker_type = st.radio(
        "Select Reranker type:", ["None", "Cross-Encoder", "Colbert"]
    )
    print_log(f"User selected reranker type: {reranker_type}")

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        print_log(f"User asked: '{user_input}'")
        with st.spinner("Processing..."):
            print_log(
                f"Getting answer from assistant using {model_choice} model and {search_type} search and {reranker_type} as Reranker"
            )
            answer_data = generate(user_input, model_choice, search_type, reranker_type)
            st.success("Completed!")
            st.write(answer_data["answer"])

            # Display monitoring information
            st.write(f"Response time: {answer_data['total_response_time']:.2f} seconds")
            st.write(f"Model used: {answer_data['model_used']}")
            st.write(f"Total tokens: {answer_data['total_tokens']}")

            # Save conversation to database
            print_log("Saving conversation to database")
            save_conversation(
                st.session_state.conversation_id,
                user_input,
                answer_data,
            )
            print_log("Conversation saved successfully")

            with st.form("fb_form"):
                st.radio(
                    "feedback thumb",
                    options=["👍", "👎"],
                    key="thumbs",
                    horizontal=True,
                )

                st.text_input("feedback comment", key="comment")
                st.form_submit_button("Send", on_click=feedback_cb)


if __name__ == "__main__":
    print_log("Starting Application...")
    main()
