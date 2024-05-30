from transformers import pipeline

# Mask Filling : FIll the blank
unmasker = pipeline("fill-mask")
output = unmasker("This is the <mask> opportunity to study at this top university in the united state.", top_k=2)
print(output)