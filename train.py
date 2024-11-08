import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# OverallQual : 주택의 전체 품질 평가
# GrLivArea : 주택의 총 주거 면적
# YearBuilt : 주택의 건축 연도
X = df[['OverallQual','GrLivArea','YearBuilt']]
y = df['SalePrice']

model = LinearRegression(fit_intercept=True)
model.fit(X.values, y.values)

if input('auto predict? (y/n): ') == 'y':
    test_X = test_df[['OverallQual','GrLivArea','YearBuilt']]
    results = model.predict(test_X.values)
    # 각 변수별 산점도
    plt.subplot(3, 1, 1)
    plt.scatter(test_df['OverallQual'], results, alpha=0.5)
    plt.title('Overall Quality vs Sale Price')
    plt.xlabel('Overall Quality')
    plt.ylabel('Sale Price')

    plt.subplot(3, 1, 2)
    plt.scatter(test_df['GrLivArea'], results, alpha=0.5)
    plt.title('Ground Living Area vs Sale Price')
    plt.xlabel('GrLivArea')
    plt.ylabel('Sale Price')

    plt.subplot(3, 1, 3)
    plt.scatter(test_df['YearBuilt'], results, alpha=0.5)
    plt.title('Year Built vs Sale Price')
    plt.xlabel('YearBuilt')
    plt.ylabel('Sale Price')

    plt.tight_layout()
    plt.show()
else:
    quality = int(input('Overall Quality: '))
    area = int(input('Ground Living Area: '))
    year = int(input('Year Built: '))
    print(model.predict([[quality, area, year]]))