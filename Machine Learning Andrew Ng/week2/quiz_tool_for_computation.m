
give_matrix = input("please type the matrix ")

target_element_row = input("please enter the row of target feature ")
target_element_column = input("please enter the target column ")

meanScore = mean(give_matrix)
minScores = min(give_matrix)
maxScores = max(give_matrix)

columnRange = maxScores(target_element_column) - minScores(target_element_column)



currentFeatureValue = give_matrix(target_element_row, target_element_column)

normalizedFeatureValue = (currentFeatureValue - meanScore(target_element_column)) / columnRange