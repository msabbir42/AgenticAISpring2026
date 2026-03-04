# Topic 1: Running an LLM

## Notebooks:

[Topic_1.ipynb](https://github.com/msabbir42/AgenticAISpring2026/blob/b92d449b715bcd8c2d49cd19ce7c8fec42901468/Topic%201/Topic_1.ipynb): This notebook contains outlines where I created separate sections for each task. At the begining of the tasks, I mentioned the discussion (if there was anything asked to be done. For example, at the begining of task 6, I compared the results based on the figures generated for that task).

The Python files are named by task. If there is no seperate python file for a task, code for that task is implemented inside the notebook.

[Files in Colab(Including Notebook)](https://github.com/msabbir42/AgenticAISpring2026/tree/b92d449b715bcd8c2d49cd19ce7c8fec42901468/Topic%201/Files%20in%20Colab(Including%20Notebook)): The folder contains the files used and generated in Colab. Specifically, [Topic_1_in_Colab.ipynb](https://github.com/msabbir42/AgenticAISpring2026/blob/b92d449b715bcd8c2d49cd19ce7c8fec42901468/Topic%201/Files%20in%20Colab(Including%20Notebook)/Topic_1_in_Colab.ipynb) contains the notebook I ran in Colab and the associated findings. While checking "Topic_1_in_Colab.ipynb" notebook, GitHub may show "Invalid Notebook". If that is the case for you, kindly download the notebook and check in VS Code (I checked and found it works).

# NOTE: 

For figures and details, please check the notebook.

# Task 4
I am using Mac and hence, did not work on 4 and 8 bit quantization.

Clearly, the GPU usage takes least time, while CPU+quantization takes most time.

# Task 5
Added timing information (please, see the findings of Task 6. also, you may check the file "llama_3.2_1b_mmlu_results_4bit_20260301_150438.json"). [please, check the notebook]
Added option to print out each question, the answer the model gives, and whether the answer is right or wrong (by  ✅, ❌). [please, check the notebook]

# Task 6

Here, I run these 3 models: "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-2b", and "Qwen/Qwen2.5-0.5B".

Clearly, LLMa and Qwen has higher performance than Gemma. However, Qwen takes significantly higher time than the Gemma and LLMa in terms of real-time, GPU time, and CPU time.
Besides, it appears that both LLMa and Qwen makes significantly less mistakes than Gemma in almost all subjects.
On the other hand, though Llama and Qwen have very similar performance, Llama is significantly faster.

It appears that though Llama has a pretty good performance, it makes more mistakes in college physics.

The mistakes do not appear to be random. 

# Task 7

I have run the codes in Google Colab and all findings are available in Topic_1_in_Colab.ipynb (inside the Files in Colab(Including Notebook)).
There, I run the following 6 models

"meta-llama/Llama-3.2-1B-Instruct",
"mistralai/Mistral-7B-v0.1",
"Qwen/Qwen2.5-7B-Instruct",
"allenai/OLMo-2-1124-7B",
"google/gemma-2b",
"Qwen/Qwen2.5-0.5B".DS_Store


A surpsing thing I noticed in colab is CPU+ no quantization took max time which is surprising to me, since in mac the max time was consumed by CPU+quantization.

