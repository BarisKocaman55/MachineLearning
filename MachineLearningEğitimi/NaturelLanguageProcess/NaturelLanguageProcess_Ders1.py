import numpy as np
import pandas as pd
import re

comments = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)


comment = []
size = int(comments.size / 2)
for i in range(size - 1):
    comment[i] = re.sub('[^a-zA-Z]', ' ', comments['Review'][i])
    comment[i] = comment[i].lower()