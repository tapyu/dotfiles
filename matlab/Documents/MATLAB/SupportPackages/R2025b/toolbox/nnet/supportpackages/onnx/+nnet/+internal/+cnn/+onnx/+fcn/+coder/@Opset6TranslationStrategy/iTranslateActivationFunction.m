function s = iTranslateActivationFunction(~, s)
%   Copyright 2024 The MathWorks, Inc.

switch s
    case 'Sigmoid'
        s = 'sigmoid';
    case 'Softsign'
        s = 'softsign';
    case 'Relu'
        s = 'relu';
    case 'Tanh'
        s = 'tanh';
    case 'HardSigmoid'
        s = 'hard-sigmoid';
    otherwise
        % Maintain name for error message
end
end

