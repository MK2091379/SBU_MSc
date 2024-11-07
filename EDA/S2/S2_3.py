from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text
text = "Python is an amazing programming language. It is widely used for data analysis, web development, automation, and more."

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()