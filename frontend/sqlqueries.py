create_conv_table_query = """
    CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
    """

create_feedback_table_query = """
    CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
    """

insert_conv_query = """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, prompt_tokens, completion_tokens, total_tokens, 
                timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
            """

insert_feedback_query = "INSERT INTO feedback (conversation_id, feedback, comment, timestamp) VALUES (%s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))"
