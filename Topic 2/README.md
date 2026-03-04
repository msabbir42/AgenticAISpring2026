# Topic 2: Agent Frameworks

## Notebooks:

[Topic_2.ipynb](https://github.com/msabbir42/AgenticAISpring2026/blob/main/Topic%202/Topic_2.ipynb): This notebook contains outlines where I created separate sections for each task. At the begining of the tasks, I mentioned the discussion (if there was anything asked to be done). 

The Python files are named by task.

# NOTE:
For details, please check the notebook.

# Task 1

Task:
In the task, it was asked to "Modify the code so that if the input is the word "verbose", then each node prints tracing information to stdout, and if the input is "quiet", the tracing information is not printed."

Here with the verbose mode, the code printed the input and output along with traces where (i.e., the function) it is.
The code simply compares whether the input is "verbose" and sets the environment variable "bool_verbose" accordingly.


Note:
While working on this task, I noticed that I get multiple responses (e.g., User: .... Assistant: ... User: .... Assistant: ...) though I only ask for once. 
I modified the code to only return the part of the response that corresponds to the current prompt, and ignore any subsequent inputs that can be included in the LLM's response.
The modification is made inside call_llm() function in langgraph_simple_llama_agent.py, where I check if the response contains "\nUser:" and if so, I split the response at that point and only keep the first part. 
This way, I can ensure that I only get the LLM's response to the current prompt and ignore any additional user inputs that may be included in the response.


# Task 2

To understand, what happens when the program is run with an empty input, I used the given code in the Topic.
As explained at the beginning of the task 1, there is a problem in the given code I think. Since, the code automatically runs more than once when I ask only once (see the output of the cell). I fixed the problem above and below, except for the first part of task 2 (since the primary purpose is to understand how it works with an empty input only).

My understanding: 
I provided empty input three times and I found random answer each time. 
I think the reason is that the LLM is generating a response based on the empty input, 
and since there is no specific context or information provided, it can generate a wide range of responses.

This says that less large and sophisticated LLMs are not enough intelligent enough with such a basic input.

# Task 3

As can be seen in the traces and output of the cell, the code runs both LLMs parallely and responds.
The models are parallelled by this statement graph_builder.add_edge(["call_llama", "call_qwen"], "print_both_responses")


# Task 4

As can be seen in the traces and output of the cell, the code first asks LLama. If the user specifically mentions "Hey Qwen", it asks Qwen. Otherwise, it only asks LLama.


# Task 5

Message API is included and Qwen is disabled.


# Task 6

As can be seen in the traces and output of the cell, even without mentioning specific question to Qwen 
and simply asking "Hey Qwen, what do you think?" answers about the Dhaka city of Bangladesh which I originally asked Llama.
This shows that the code can handle the assigned task.

However, I would like to note a potential issue. The Qwen includes several/many repetitiive phrases at the end of the response.
I tried to fix it in several ways and spent significant time only for this. For instance, setting top_p=0.95, enabling sampling, trying with -Instruct version, and following several other approaches while chatting with Gemini.
However, it is not fixed. One possible explanation is that perhaps, it is due to using a very small Qwen model (Qwen2.5-0.5B) which may not be able to wisely handle the such complex conversation and it may be generating repetitiive phrases as a result.
The reason for this explanation is that for any of the early tasks, I did not observe such repetitiive phrases in the response of Qwen.

To fix it, I also tried with a bit larger model Qwen2.5-1.5B. However, the same issue is observed.
In my laptop, I could not run Qwen2.5-0.5B-Intruct model, since it causes run time crash. I tried to fix several ways, but those did not work out.
But I hope this should not be a big issue, since I was able to understand the intended task.


# Task 7

To understand whether the code works perfectly for the crash recovery, first I asked it a question about Dhaka city. Then, I forcefully stopped by pressing Restart.
Later (please, see the output of next cell), I asked "What did I ask you before? I asked about a city."). It was able to answer that I asked about Dhaka city.
The reason it was able to answer is that it saved the data in "chat_history.db" file and so, I was able to start the chat where I left off.
Inside the main function of task_7.py, with "with SqliteSaver.from_conn_string("chat_history.db") as memory_saver:", the code creates a connection to the SQLite database file "chat_history.db" and uses it as a memory saver to store the chat history.


