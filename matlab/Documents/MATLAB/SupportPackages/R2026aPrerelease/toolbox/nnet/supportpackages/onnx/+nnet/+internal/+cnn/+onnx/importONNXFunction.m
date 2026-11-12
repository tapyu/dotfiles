function params = importONNXFunction(modelfile, userOutputFcnName)

%   Copyright 2020-2025 The MathWorks, Inc.

% Internal Settings
maxNameLength = 40;
maxSupportedOpset = 20;
minSupportedOpset = 6;
nnet.internal.cnn.onnx.setAdditionalResourceLocation();     % For SPKG resource catalog.
[inputFilename, outputFcnPath, outputFcnName] = iValidateInputs(modelfile, userOutputFcnName);
% Deserialize the model from protobuf into MATLAB
modelProto       = nnet.internal.cnn.onnx.ModelProto(inputFilename);
% Check if the model IR is fully supported
nnet.internal.cnn.onnx.warnIfFutureIR(modelProto);
% Check that the model is nonempty
iCheckEmptyModel(modelProto,inputFilename);
% Clear the persistent variable used to generate new names
import nnet.internal.cnn.onnx.fcn.*
clear uniqueName; 

modelTranslation = nnet.internal.cnn.onnx.fcn.ModelTranslation(modelProto, outputFcnPath, outputFcnName, maxNameLength, '', false);
% Check if the model opset version is fully supported
iValidateOpsetVersion(modelTranslation.OpsetVersion, maxSupportedOpset, minSupportedOpset);
params = modelTranslation.Params;

% Write the code to the m-file
writeOutputFile(modelTranslation);
if isempty(outputFcnPath)
    fileString = [outputFcnName '.m'];
    fprintf('%s\n', getString(message('nnet_cnn_onnx:onnx:IOFMessageLine1', fileString)));
    fprintf('%s\n', getString(message('nnet_cnn_onnx:onnx:IOFMessageLine2', outputFcnName)));
else
    fileString = fullfile(outputFcnPath, [outputFcnName '.m']);
    fprintf('%s\n', getString(message('nnet_cnn_onnx:onnx:IOFMessageLine1', fileString)));
    fprintf('%s\n', getString(message('nnet_cnn_onnx:onnx:IOFMessageLine2a', outputFcnPath, outputFcnName)));
end
% Report any node translation issues
issues = modelTranslation.ToplevelGraphTranslation.TranslationIssues;
iWarnIfTranslationIssues(issues);
end

function [inputFilename, outputFcnPath, outputFcnName] = iValidateInputs(userInputFilename, userOutputFcnName)
inputFilename = iValidateInputFile(userInputFilename);
% Validate output filename and path
if ~(ischar(userOutputFcnName) || (isstring(userOutputFcnName) && isscalar(userOutputFcnName)))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FirstArgString')));
end
userOutputFcnName = char(userOutputFcnName);
[outputFcnPath, outputFcnName, ext] = fileparts(userOutputFcnName);
if ~isempty(outputFcnPath) && ~exist(outputFcnPath, 'dir')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:OutputFilepathDoesNotExist', outputFcnPath)));
end
if ~isempty(ext) && ~isequal(ext, '.m')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:OutputExtensionNotM')));
end
if ~isvarname(outputFcnName)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidFcnName')));
end
end

function Filepath = iValidateInputFile(Filename)
if ~(ischar(Filename) || (isstring(Filename) && isscalar(Filename)))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FirstArgString')));
end
% Check if the file extension is valid (i.e., if the filename ends with '.onnx')
[~,~,ext] = fileparts(Filename);
if (~strcmp(char(ext),'.onnx'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidFileExtension', Filename)));
end
Filepath = which(char(Filename));
if isempty(Filepath) && exist(Filename, 'file')
    Filepath = char(Filename);
end
if ~exist(Filepath, 'file')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FileNotFound', Filename)));
end
end
function iCheckEmptyModel(modelProto, filename)
% Check if a layer graph is present in the imported model
    if isempty(modelProto.graph)
        throwAsCaller(MException(message('nnet_cnn_onnx:onnx:EmptyModel',filename)));
    end
end

function iWarnIfTranslationIssues(translationIssues)
[issueCounts, issueStrings] = iParseTranslationIssues(translationIssues);
% Warn if there are unsupported or missing layers
iSummarizeTranslationIssues(issueCounts, issueStrings);
end

function [IssueCounts, IssueStrings] = iParseTranslationIssues(translationIssues)
% Summarizes the issues encountered during translation
% Report on issues that led to Placeholder creation as well as warnings.
allIssues       = arrayfun(@(x) getMessageString(x), translationIssues, 'UniformOutput', false);
IssueStrings    = string(unique(allIssues));

IssueCounts = zeros(numel(IssueStrings), 1);
for i=1:numel(IssueStrings)
    IssueCounts(i) = sum(strcmp(IssueStrings(i), allIssues));
end
end

function iSummarizeTranslationIssues(issueCounts, issueStrings)
% Error if any unsupported layers, refer user to importONNXLayers
if ~isempty(issueCounts)
    prologue = message('nnet_cnn_onnx:onnx:IOFTranslationIssues').getString();
    
    unsupportedLayersString = "";
    for i=1:numel(issueCounts)
        unsupportedLayersString = join([unsupportedLayersString,...
            sprintf('%d operator(s)\t:\t%s', issueCounts(i), issueStrings(i))], newline);            
    end
    
    epilogue = "";
    
    nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:IssuesDuringImport', prologue, unsupportedLayersString, epilogue);
end

end

function iValidateOpsetVersion(OpsetVersion, maxSupportedOpset, minSupportedOpset)
if OpsetVersion > maxSupportedOpset || OpsetVersion < minSupportedOpset
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:OpsetVersionUnsupportedForImport', num2str(OpsetVersion), num2str(maxSupportedOpset), num2str(minSupportedOpset));
end
end
