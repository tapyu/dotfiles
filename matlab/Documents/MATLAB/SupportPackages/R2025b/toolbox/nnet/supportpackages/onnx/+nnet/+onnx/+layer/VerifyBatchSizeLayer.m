classdef VerifyBatchSizeLayer < nnet.layer.Layer & nnet.layer.Formattable ...
        & nnet.internal.cnn.layer.Traceable
    % VerifyBatchSizeLayer   Verify fixed batch size
    %
    %   layer = VerifyBatchSizeLayer(name, requiredBatchSize, batchDim,
    %   inputLayerName) creates a layer that verifies that dimension
    %   batchDim of its input has size requiredBatchSize.

    %   Copyright 2021-2024 The MathWorks, Inc.
    properties
        RequiredBatchSize
        BatchDim
        InputLayerName
    end

    methods
        function this = VerifyBatchSizeLayer(name, requiredBatchSize, batchDim, inputLayerName)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:VerifyBatchSizeLayerDescription', requiredBatchSize));
            this.Type = getString(message('nnet_cnn_onnx:onnx:VerifyBatchSizeLayerType'));
            this.RequiredBatchSize = requiredBatchSize;
            this.BatchDim = batchDim;
            this.InputLayerName = inputLayerName;
        end

        function X = predict( this, X )
            if size(X, this.BatchDim) ~= this.RequiredBatchSize
                error(message('nnet_cnn_onnx:onnx:WrongFixedBatchSize', ...
                    this.InputLayerName, this.BatchDim, this.RequiredBatchSize));
            end
        end
    end
end
