import sys

print (f"Hello, ML Engineer! Running Python {sys.version_info.major}.{sys.version_info.minor}")

salary = 165_000
accuracy = 0.9234
model_name = "fraud_detector"
is_deployed = True
last_score = None

print (type(salary), type(accuracy), type(model_name), type(is_deployed), type(last_score))

raw_age = "34"
age = int(raw_age)
price = float("19.99")
print (age+1, price*2)

name, acc, revenue = "churn-model-v2", 0.98765, 1234567.891

print (f"{name} accuracy: {acc:.2%}")
print (f"Revenue: ${revenue:,.2f}")
print (f"[{name:>20}]")

raw = " Fraud Detector-V2 "
clean = raw.strip().lower()
parts = clean.split("-")
slug = "_".join(parts)
print (clean,parts,slug)

scores = [88,92,79,95,61]
scores.append(73)
scores.sort(reverse=True)
print("Sorted:",scores)

print("Top 3:", scores[:3])
print("Last:", scores[-1])
print("Every other:", scores[::2])
print("Mean:", sum(scores)/len(scores))

passed = []
for s in scores:
    if s >= 75:
        passed.append(s)

print("Passed (loop):", passed)

passed = [s for s in scores if s >= 75]
curved = [min(s+5,100) for s in scores]
labels = ["pass" if s >= 75 else "fail" for s in scores]
print(passed,curved,labels,sep="\n")

names = ["Ava","Ben","Cara"]
years = [4,1,6]
for rank, (person,yoe) in enumerate(zip(names, years), start=1):
    print (f"#{rank} {person}: {yoe} yrs experience")
    
from collections import namedtuple

point = (37.77,-122.42)

lat,lon = point

Employee = namedtuple("Employee", ["name","title","salary"])
e = Employee("Ava","ML Engineer",165_000)
print (f"{e.name} is a {e.title} making ${e.salary:,}")


salary_by_city = {
    "San Francisco": 190_000,
    "Seattle": 175_000,
    "New York": 180_000,
    "Austin": 150_000
}
salary_by_city["Boston"] = 170_000
print(salary_by_city["Seattle"])
print(salary_by_city.get("Denver", "N/A"))

for city, pay in sorted(salary_by_city.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{city: <14}: ${pay:,}")
    
high_pay = {c: p for c, p in salary_by_city.items() if p >= 175_000}
print(high_pay)

from collections import defaultdict
skills = [("python", "Ava"), ("sql", "Ben"), ("python", "Cara"), ("pytorch", "Ava")]
people_by_skill = defaultdict(list)
for skill, person in skills:
    people_by_skill[skill].append(person)
print(dict(people_by_skill))

from collections import Counter
words = "The model the data the pipeline data model the".split()
counts = Counter(words)
print(counts.most_common(2))
print(counts["data"], counts["missing"])

job_a = {"python","sql","pytorch","aws","docker"}
job_b = {"python","sql","spark","airflow","aws"}
print("Common:", sorted(job_a & job_b))
print("Only in A:", sorted(job_a - job_b))
print("All:", sorted(job_a | job_b))


import timeit
big_list = list(range(100_000))
big_set = set(big_list)
t_list = timeit.timeit(lambda: 99_999 in big_list, number=200)
t_set=timeit.timeit(lambda: 99_999 in big_set, number=200)
print(f"list: {t_list:.4f}s set: {t_set:.6f}s")

from collections import deque

window = deque(maxlen=3)
for x in [10,20,30,40,50]:
    window.append(x)
    print(f"window = {list(window)} moving avg = {sum(window)/len(window):.1f}")
    
import heapq
model_scores = {"xgboost": 0.91, "logreg": 0.84, "rf": 0.89, "mlp": 0.87, "svm": 0.82}
top3 = heapq.nlargest(3, model_scores.items(), key=lambda kv: kv[1])
print(top3)

import bisect
thresholds = [580,670, 740,800]
bands = ["Poor","Fair","Good","Very Good","Exceptional"]

def fico_band(score: int) -> str:
    return bands[bisect.bisect_right(thresholds, score)]

for s in (550,690,810):
    print (f"Fico {s} -> {fico_band(s)}")
    
def bad_append(item, bucket=[]):
    bucket.append(item)
    return bucket

print(bad_append(1), bad_append(2))

def good_append(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

print(good_append(1), good_append(2))

a = [[0] * 3] * 3
a[0][0] = 9
print(a)

b = [[0] * 3 for _ in range(3)]
b[0][0] = 9
print(b)

x = [1, 2]
y = [1, 2]
print(x == y, x is y)





def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return seen[target - value], i
        seen[value] = i
    return None

def is_anagram(word1, word2):
    return Counter(word1) == Counter(word2)

top2 = [skill for skill, _ in Counter(posts).most_common(2)]
