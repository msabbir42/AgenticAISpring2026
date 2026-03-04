# Exercise 1

1. The model without RAG hallucinates significantly. For instance, the model invented an "ABA meeting in Washington D.C." where Mr. Flood excused the Mayor's absence. As far as I understood, the truth (found via RAG) is that Flood praised him in Congress for his service.

2. Yes, apparently, RAG ground the answers in the actual manual. The RAG answers contain direct quotes (e.g., "ignored repeated lawful orders") and specific technical details (e.g., "0.74 inches") that appear in the retrieved chunks. Besides, the Congressional Record questions refer to January 2026. The base model cannot possibly know these future events, so the fact that the RAG system answers them with specific dates and names proves it is reading the documents.

3. Yes! Without RAG (Model's Knowledge): It said (in response to "What is the correct spark plug gap for a Model T Ford?") .035 inches which is correct or very close to the correct answer 0.025 to 0.030 inches. On the other hand, with RAG: It said 0.74 inches and quoted the text "should be 74".


# Exercise 2

1. Yes, GPT 4o is much better in avoiding hallucinations. For instance, GPT 4o honestly said it does not know for the following question, whereas Qwen 2.5 hallucinated.

Question: What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?
------------------------------------------------------------
GPT 4o: I'm sorry, but I do not have information about events or statements made after October 2023, including anything that may have occurred on January 13, 2026.

2. All Model T Questions: Spark plug gap (0.025-0.030"), oil type (SAE 30), and transmission band adjustments. It has this "world knowledge" internally.

Data Age vs. Training Cutoff:
    GPT-4o Mini Cutoff: October 2023.
    Model T Corpus: ~1919–1927. (Pre-dates cutoff by ~100 years → INCLUDED in training).
    Congressional Record Corpus: January 2026. (Post-dates cutoff by >2 years → EXCLUDED from training).

Result: It answers Model T questions perfectly from memory but correctly refuses or hallucinates on the future Congressional Record questions.


# Exercise 3

I used GPT 5.2 (Thinking) for comparision. Here are the outputs: https://chatgpt.com/share/69a7ce2d-c748-8012-a29e-57d0856b3b23

* **Where does the frontier model's general knowledge succeed?**
The frontier model (GPT 5.2) succeeds flawlessly on the historical and technical Model T questions. It accurately provides the 1/32-inch spark plug gap, the correct oil type, and carburetor adjustments purely from its vast pre-trained internal knowledge base, without needing any document uploads.

* **When did the frontier model appear to be using live web search?**
It appeared to use live web search for the highly specific, recent (or hypothetical future) 2026 Congressional Record queries.

* **Where does your RAG system provide more accurate, specific answers?**
While the frontier model outperformed it overall, the RAG system significantly improved the local Qwen model's accuracy on the Congressional Record queries compared to its baseline.

* **What does this tell you about when RAG adds value vs. when a powerful model suffices?**
A powerful frontier model suffices (and often performs better) for well-documented historical facts, general knowledge, or public data it can search. RAG adds the most value when there is a need to force a smaller, offline, or less capable model to answer questions based strictly on proprietary, private, or highly domain-specific documents that a cloud model cannot access.


# Exercise 4

1. When does adding context stop helping? It stops helping once the correct answer is found.

Example: For the Carburetor question, k=1 gave the full correct procedure. Increasing it to 3, 5, or 10 just repeated the same information, wasting computing power without improving the answer.

2. When does too much context hurt? It hurts when it retrieves similar-sounding but wrong information, confusing the model.

Example: For the Oil question, the system retrieved maintenance manuals for aircraft (mentioning "Learjet" and "MIL-L-7870" grease) instead of the Model T. The model got confused and recommended airplane grease for a car engine.

3. How does k interact with chunk size? They work like a see-saw (inversely related):

Small Chunks: A higher k (more pieces) is needed to puzzle together the full story.
Large Chunks: Alower k (fewer pieces) is needed, otherwise the model can be flooded with too much irrelevant text.


# Exercise 5
1. Yes, it knows. For instance, tt was able to say the capital of France correctly. 

2. Yes, the model hallucinates a lot. For instance, though the model was able to say the capital of France correctly, it provided lots of information along with that for such a simple question. Similar problem occurred for the other questions.

3. Apparently, it helps.


# Exercise 6

1. Which phrasings retrieve the best chunks? 

It seems like phrasings that used simple, direct language like "How often..." and "How frequently..." worked better.

2. Do keyword-style queries work better or worse? 

Worse. The keyword-style query "Engine service frequency guidelines" appears to be failed the hardest. It retrieved the EU AI Act (irrelevant legal text) because it latched onto abstract words like "frequency" and "guidelines" rather than the semantic concept of car maintenance.

3. What does this tell you about query rewriting? 

Retrieval is extremely brittle. Changing the wording slightly can result in 0% overlap in results (as seen in 4 out of 5 comparisons).


# Exercise 7

Overlap 0 (No overlap): The text was chopped right in the middle of a thought. The AI found the second half of the instructions, but completely missed the first half. It gave an incomplete answer.

Overlap 64: Because the overlap was too small, the true answer got chopped into unreadable pieces. Maybe, the AI got desperately confused.

Overlap 128 (The Sweet Spot): The strips of paper overlapped just enough to act like "tape." The AI read it clearly and gave a flawless 5-step answer.

Overlap 256: This overlap was too big (half the size of the chunk itself). It fed the AI massive amounts of duplicate, repeating text. The AI got overwhelmed.


# Exercise 8

1. How does chunk size affect retrieval precision?

    128 (Too Small): Poor precision. It pulls broken, useless fragments (e.g., pulling "engine (a metal cap covers it)" when asked about oil).

    512 (Balanced): High precision. It retrieves exact, relevant paragraphs without extra clutter.

    2048 (Too Large): Low precision. It pulls massive walls of text containing irrelevant chapters (e.g., retrieving "Bendix Shaft" instructions when asked "How do I adjust the carburetor on a Model T?").

2. How does it affect answer completeness?

    128: Answers are highly incomplete and force the AI to guess or hallucinate (e.g., guessing the spark plug gap is just "close").

    512: Answers are highly complete, perfectly capturing multi-step instructions (e.g., giving a flawless 5-step answer for the carburetor rod).

    2048: Answers become too wordy, and the AI actually gets "lost" in the massive text.

3. Is there a sweet spot for your corpus?

    Yes: 512 characters. It is large enough to keep full procedures (like the 5 steps for the carburetor) intact, but small enough to keep the AI focused without getting distracted by surrounding chapters.

4. Does optimal size depend on the type of question?

    Yes. * Simple Fact Questions (e.g., "What oil should I use?") may sometimes survive smaller chunks if the keyword is right next to the answer.


# Exercise 9

**1. When is there a clear "winner"?**
A clear winner happens when a query asks for highly specific, unique terms that only appear in one specific place.

* **Example:** "How to install a Bendix shaft?" has the largest gap (**0.071**) between the first and second chunk. The top score (**0.689**) shows the system easily found the exact, unique paragraph detailing this specific part.

**2. When are scores tightly clustered (ambiguous)?**
Scores cluster tightly when a topic is broad and mentioned repeatedly across many different pages.

* **Example:** "How do I adjust the carburetor...?" has a microscopic gap of just **0.001**. The top scores (0.627 and 0.625) are nearly identical because a car manual discusses the carburetor everywhere, making it hard for the search engine to decide which paragraph is the absolute "best."

**3. What score threshold filters out irrelevant results?**
A threshold around **0.40** is the sweet spot for this specific pipeline.

* **Example:** The completely off-topic trick question ("What is the CEO's favorite color?") maxed out at a score of **0.354**. This proves that any chunk scoring in the 0.30s is irrelevant junk and should be filtered out to prevent the AI from getting confused.

**4. How does score distribution correlate with answer quality?**

* **High Scores (0.60+):** The AI receives highly relevant context (like the Mayor David Black chunk at 0.641) and will generate a perfectly factual answer.
* **Low Scores (< 0.40):** The AI is fed irrelevant garbage and will either hallucinate an answer or be forced to say "I don't know."

**5. How did the 0.60 threshold experiment affect results?**
The 0.60 threshold seems to be too strict. It destroyed the pipeline by filtering out 100% of the chunks for 6 out of the 10 questions.

* **Example:** "What is the correct spark plug gap?" is a perfectly valid question, but its top chunk only scored **0.587**. The > 0.60 filter threw away the correct answer, which would cause the AI to fail a question it actually had the information for.


# Exercise 10

1. Which prompt produces the most accurate answers? 
    Strict Grounding and Structured Output.

    Why: For the Oil question (where the retrieval failed and returned Aircraft manuals), Strict Grounding correctly said "I don't know," preventing a hallucination. Structured Output went even further, explicitly noting: *"The question seems to be asking about a Model T... but the context appears to be related to aircraft."

2. Which produces the most useful answers? 
    Structured Output.

    Why: By forcing the model to list "Relevant Facts" before answering, it "showed its work." This step allowed it to catch the error where the retrieved text was about Learjets, not Model Ts, providing a warning instead of bad advice.

3. Is there a trade-off between strict grounding and helpfulness? 

    Yes.

    It seems permissive was "helpful" but wrong.

    Strict Grounding was "unhelpful" (gave no answer) but safe: It refused to give advice based on the wrong manual.


# Exercise 10
1. Can the model successfully combine information from multiple chunks? 
    Yes.

    Evidence: In the "slow speed vs. brake band" question, the model successfully synthesized a comparison. 

2. Does it miss information that wasn't retrieved? 
    Yes, significantly.

    Evidence: For the "carburetor overhaul" question, the model only gave a high-level summary (1. Remove, 2. Overhaul, 3. Install).

3. Does contradictory information cause problems? 
    It causes "Silence" rather than confusion here.

    Evidence: For the "safety warnings" question, the model confidently stated "There are no explicit safety warnings." This is false (a manual would have many), but because the warnings are scattered across hundreds of pages and didn't appear in the Top-10 chunks, the model assumed they didn't exist. This is a dangerous failure mode.



# Acknowledgement
After working on this topic, I understand how does RAG actually work, how can we make bettee some LLMs using RAG, when it can work better than state of the art models, what are the factors (e.g., prompt template variation, parapharsing, chunk size, chunk overlap) that can affect the RAG significantly, etc, though I am new to LLM. However, I acknowledge that I used GenAI very heavily (Gemini 3.1 Pro; GPT 5.2 (Plus user)) for this topic, since there were many tasks and I worked on all. However, I tried to understand the core things. Besides, I acknowledge that I used GenAI heavily for analyzing the outputs as well, since the outputs were so large. Though I used GenAI to analyze the results and aswer the asked questions, I tried to understand as much as possible (e.g., compared with several outputs with statements of GenAI for each task for verification). Besides, I spent more than 6 hours only for this Topic. Also, I do not have any knowledge on Model T service manual and Congressional Record.

To be transparent, I am adding the prompts:
1. https://gemini.google.com/share/622baff013bb
2. I used Colab integrated Gemini as well. However, I do not see any way to share them. Happy to provide screenshots if you want.
3. https://chatgpt.com/share/69a7ce2d-c748-8012-a29e-57d0856b3b23