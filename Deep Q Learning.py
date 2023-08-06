"""
Reward:
-eat food: +10
-game over: -10
-else: 0

Action:
[1, 0, 0] -> Straight 
[0, 1, 0] -> Right turn 
[0, 0, 1] -> Left turn

State(11 values):
[danger straight, danger left, danger right,
direction left, direction right,
direction down, direction up
food left, food right,
food down, food up]

Model:
state --> deep neural net -> action

(Deep) Q Learning
Q value = Quality of action

0. Init Q Value (= init model)
1. Choose action (model.predict(state)) (or random move)
2. Perform action
3. Measure reward
4. Update Q value (+train mode)

Bellman equation:


"""