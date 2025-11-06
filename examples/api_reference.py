import tunecontrol as tc

print('\n'.join(tc.tasks.list_task_names()))  # list available tasks
task = tc.make("cartpole/2d/mae/deterministic")

# example evaluate the middle of the domain
theta = (task.bounds[0] + task.bounds[1]) / 2
value, info = task.evaluate(theta)
print(value.item())
