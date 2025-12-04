from sklearn.model_selection import train_test_split
from load_dataset import load_data

X, y, classes = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
