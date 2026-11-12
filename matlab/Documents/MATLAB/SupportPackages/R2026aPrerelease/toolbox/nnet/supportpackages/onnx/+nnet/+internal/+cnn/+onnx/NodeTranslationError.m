classdef NodeTranslationError < nnet.internal.cnn.onnx.NodeTranslationIssue
    % A mesage describing an error that occurred during translation of an
    % ONNX network
    
    %   Copyright 2020 The MathWorks, Inc.    

    methods
        function obj = NodeTranslationError(node, msg)
            obj.LayerName = string(node.name);
            obj.Operator = string(node.op_type);
            obj.Message = msg;
        end

    end
end

