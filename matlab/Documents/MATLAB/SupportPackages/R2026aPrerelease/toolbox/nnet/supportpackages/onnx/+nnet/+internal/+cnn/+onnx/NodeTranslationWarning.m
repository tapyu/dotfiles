classdef NodeTranslationWarning < nnet.internal.cnn.onnx.NodeTranslationIssue
    % A mesage describing an issue that occurred during translation of an
    % ONNX network, leading to a warning. 
    
    %   Copyright 2020 The MathWorks, Inc.    

    methods
        function obj = NodeTranslationWarning(node, msg)
            obj.LayerName = string(node.name);
            obj.Operator = string(node.op_type);
            obj.Message = msg;
        end

    end
end

