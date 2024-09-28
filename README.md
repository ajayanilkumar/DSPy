
# Signatures

A signature is a declarative specification of input/output behavior of a DSPy module.
- Tell the LLM what it needs to do rather than specify how we should ask the LLM to do it.
- It makes the code more modular and clean.
- LLM calls can be optimized into high quality prompts. No need to do trial and error with prompt templates.

### Inline Signatures

Can be defined as short strings with argument names that define semantic roles for inputs and outputs.

```
Question Answering : "question -> answer"

Sentiment Classification : "sentence -> sentiment"

Summarization : "document -> summary"
```

There can also be multiple input/output fields.

```
RAG QA : "context, question -> answer"
```

### Class-based DSPy Signatures

For advanced tasks you will need more verbose signatures to:
- Clarify something about the nature of the task. (expressed as docstring).
- Describe the nature of the input and output fields. (`desc` keyword in `dspy.InputField` or `dspy.OutputField`)

```python
class Emotion(dspy.Signature):
	"""Classify emotion among sadness, joy, love, anger, fear, surprise."""

	sentence = dspy.InputField()
	sentiment = dspy.OutputField()

sentence = "I started feeling a bit vulnerable when the giant spotlight started blinding me"

classify = dspy.Predict(Emotion)
classify(sentence=sentence).sentiment
```

# Modules

It is a building block from LLM programs.
- Each built in module abstracts a prompting technique (eg: ReAct or chain of thought).
- They are generalized to handle any DSPy Signature.
- A DSPy Module has learnable parameters.
- Multiple modules can be composed into bigger modules or programs.
- It is inspire by PyTorch

To use a module, we first declare it by giving it a signature. Then we call the module with the input arguments, and extract the output fields.

### Some in-built DSPy Modules

`dspy.Predict` : Basic predictor. Does not modify the signature.

`dspy.ChainOfThought` : Teaches the LM to think step-by-step before committing to the Signatures response.

`dspy.ProgramOfThought` : Teaches the LLM to output code, whose execution results will dictate the response.

`dspy.ReAct` : An agent that can use tools to implement the given signature.

`dspy.MultiChainComparison` : Can compare multiple outputs from Chain Of Thought to produce a final output/prediction.

`dspy.majority` : Can do basic voting to return the most popular response from a set of predictions.



