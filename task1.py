print("Welcome To ChatBot.\n")
exit = 1
greetings = ["hi", "hello", "hey"]
endings = ["bye", "exit", "quit"]
dialogues = {
    "how are you?": "I am fine.",
    "what is your name?": "I am ChatBot.",
    "who made you?": "Mohammad Faizan Ansari made me."
}
while exit == 1:
    user_input = input("Enter Input:~ ").lower();
    if user_input in greetings:
        print("Hello!")
    elif user_input in endings:
        print("Have Nice Day. Bye.")
        exit = 0
    elif user_input in dialogues:
        print(dialogues[user_input])
    else:
        print("Sorry ! Dialogue Not Found.")