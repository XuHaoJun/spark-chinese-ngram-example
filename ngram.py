from pyspark import SparkContext

# n=2, line="1234"; result=["12", "23", "34", "4"]
def nGramChineseSplit(n, line):
     return list(map(lambda anchorPoint: line[anchorPoint:anchorPoint + n], \
                     range(0, len(line))))

def nGram(sc, numGram):
    text_file = sc.textFile("example-data.txt")
    counts = text_file.flatMap(lambda line: nGramChineseSplit(numGram, line)) \
                .map(lambda word: (word, 1)) \
                .reduceByKey(lambda totalCount, count: totalCount + count) \
                .sortBy(lambda t: -t[1])
    return counts

sc = SparkContext("local", "N-Gram")

result = nGram(sc, 2)

result.saveAsTextFile("ngram_result")
