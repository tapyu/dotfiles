function constraints = getConverterConstraints()
% getConverterConstraints retrieve a list of constraints to check that a
% network satisfies the desired constraints for ConverterForNetwork through 
% the network analyzer. 

%   Copyright 2020 The MathWorks, Inc.

% use prediction constraint to validate weights
constraints = [
    iBuiltInConstraints
    iCustomPrediction];
end

function c = iBuiltInConstraints
c =  nnet.internal.cnn.analyzer.constraints.Constraint.getBuiltInConstraints();
end

function constraint = iCustomPrediction()
constraint = ...
    nnet.internal.cnn.analyzer.constraints.customConstraints.Prediction;
end