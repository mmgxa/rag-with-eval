import os
import psycopg2
import pytz

from sqlqueries import (
    create_conv_table_query,
    create_feedback_table_query,
    insert_conv_query,
    insert_feedback_query,
)


def print_log(message):
    print(message, flush=True)


from datetime import datetime


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "rag"),
        user=os.getenv("POSTGRES_USER", "pguser"),
        password=os.getenv("POSTGRES_PASSWORD", "pgpass"),
    )


def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:

            cur.execute(create_conv_table_query)
            cur.execute(create_feedback_table_query)

        conn.commit()
    except:
        print("couldn't create tables")
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer_data):
    timestamp = datetime.now(pytz.UTC)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                insert_conv_query,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["total_response_time"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    timestamp,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback, comment):
    timestamp = datetime.now(pytz.UTC)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                insert_feedback_query,
                (conversation_id, feedback, comment, timestamp),
            )

        conn.commit()
    finally:
        conn.close()
