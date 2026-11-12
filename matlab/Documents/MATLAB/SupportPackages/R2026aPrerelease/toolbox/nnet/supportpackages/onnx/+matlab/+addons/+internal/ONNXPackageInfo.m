classdef ONNXPackageInfo < matlab.addons.internal.SupportPackageInfoBase
    % ONNX support package support for MATLAB Compiler.
    
    %   Copyright 2020-2025 The MathWorks, Inc.
    
    methods
        function obj = ONNXPackageInfo()
            obj.baseProduct = 'Deep Learning Toolbox';
            obj.displayName = 'Deep Learning Toolbox Converter for ONNX Model Format';
            obj.name        = 'Deep Learning Toolbox Converter for ONNX Model Format';
            
            % Find the name of the webcammex file based on the platform.
            sproot = matlabshared.supportpkg.getSupportPackageRoot();
            if ispc
                onnxmexFileName = 'onnxmex.mexw64';
                onnxpbDir = fullfile(sproot, 'bin', 'win64','onnxpb.*');
            else
                macArch = computer('arch');
                if ismac
                    onnxmexFileName = "onnxmex.mex"+macArch;
                    onnxpbDir = fullfile(sproot, 'bin', macArch, 'onnxpb.*');
                else
                    onnxmexFileName = 'onnxmex.mexa64';
                    onnxpbDir = fullfile(sproot, 'bin', macArch, 'onnxpb.*');
                end
            end
            % Define all the data that should be deployed from the support
            % package. This includes the actual language data, which will
            % be archived in the CTF.
            obj.mandatoryIncludeList = {...
                onnxpbDir,...
                fullfile(sproot,'resources','nnet_cnn_onnx'),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+onnx','+layer')
                };
            obj.conditionalIncludeMap = containers.Map;
            obj.conditionalIncludeMap(fullfile(sproot, 'toolbox','nnet','supportpackages','onnx','exportONNXNetwork.m')) = ...
                {fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx','FuncName.m'),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx',onnxmexFileName)
                };
            obj.conditionalIncludeMap(fullfile(sproot, 'toolbox','nnet','supportpackages','onnx', 'importNetworkFromONNX.m')) = ...
                {fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx')
                };
            % Include mandatoryIncludeList if a ONNX layer is seen
            layerDir = fullfile(sproot, 'toolbox','nnet','supportpackages','onnx','+nnet', '+onnx','+layer');
            layers = dir(fullfile(layerDir,'*.m'));
            layerName = {layers.name};
            for i = 1:numel(layerName)
                obj.conditionalIncludeMap(fullfile(layerDir, layerName{i})) = {};
            end
        end
    end
end