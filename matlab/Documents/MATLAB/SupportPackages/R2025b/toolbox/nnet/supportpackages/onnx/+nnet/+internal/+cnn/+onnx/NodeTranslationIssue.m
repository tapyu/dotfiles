classdef NodeTranslationIssue < matlab.mixin.Heterogeneous
    % A mesage describing a problem that occurred during translation of an
    % ONNX network. 
    
    %   Copyright 2020-2022 The MathWorks, Inc.    

    properties
        LayerName
        Operator
        Message
    end
    
    methods
        function msgString = getMessageString(obj)
            msgString = getString(obj.Message);
        end
        
        function id = getMessageIdentifier(obj)
            id = obj.Message.Identifier;
        end
    end
end

