classdef NetworkMetadata
    % A class to hold ONNX network metadata
        
    % Copyright 2018-2024 The Mathworks, Inc.

    properties
        % Each comment gives the type that ONNX expects. MATLAB strings,
        % chars and any numeric type are allowed here, and will be
        % converted to the exact ONNX datatypes elsewhere when creating the
        % ModelProto. Empty values of any type will not be written to the
        % PB file.
        NetworkName     = "network";        % string
        IrVersion       = 9;                % integer
        OpsetVersion    = 8;                % integer
        ProducerName    = string(getString(message('nnet_cnn_onnx:onnx:ProducerName')));  % string
        ProducerVersion = "24.2.0";         % string. 
        Domain          = "";               % string
        ModelVersion    = [];               % integer
        DocString       = "";               % string
        % Mathworks metadata fields
        MathWorksOpsetVersion = 1;          % integer
    end
end
