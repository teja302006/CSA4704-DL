
import statistics

marks = [45, 67, 89, 72, 56, 67, 90, 38, 67, 74]

range_value = max(marks) - min(marks)
variance = statistics.pvariance(marks)
standard_deviation = statistics.pstdev(marks)
cv = (standard_deviation / statistics.mean(marks)) * 100

print("Range =", range_value)
print("Variance =", round(variance, 2))
print("Standard Deviation =", round(standard_deviation, 2))
print("Coefficient of Variation =", round(cv, 2), "%")
