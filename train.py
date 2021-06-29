from params import *

def train(generatorTrain, generatorVal):
    model=create_model()
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate), 
        metrics=['accuracy'] # métrique à changer  workforce_needed_create(1879), my_metric_fn

    )

    history=model.fit_generator(
    generatorTrain, epochs=NBEPOCH, callbacks=None,
    validation_data=generatorVal,
    class_weight=None,shuffle= SHUFFLE_DATA)
                
    model.save('checkpoint')
    return model,history