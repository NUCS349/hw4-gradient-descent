from code import GradientDescent, load_data

print('Starting example experiment')

features, targets = load_data('blobs')
learner = GradientDescent()
learner.fit(features, targets)
predictions = learner.predict(features)

print('Finished example experiment')
