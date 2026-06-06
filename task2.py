print("Spam Message Detector")
spam_keywords = [
    "win",
    "prize",
    "free",
    "offer",
    "lottery",
    "money",
    "click",
    "urgent",
    "congratulations"
]
while True:
    message = input("\nEnter Message: ").lower()
    if message == "exit":
        print("Program Closed.")
        break
    is_spam = False
    for word in spam_keywords:
        if word in message:
            is_spam = True
            break
    if is_spam:
        print("Spam Message")
    else:
        print("Not Spam Message")