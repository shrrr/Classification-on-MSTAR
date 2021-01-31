% train_path='E:\CurrentFiles\Documents\研一上课程\雷达目标识别\Target-Detection-in-MSTAR-Images-master\DataSet\train';
% test_path='E:\CurrentFiles\Documents\研一上课程\雷达目标识别\Target-Detection-in-MSTAR-Images-master\DataSet\test';

train_path = 'C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\MSTAR_BOW\preprocess_train_1';
test_path = 'C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\MSTAR_BOW\preprocess_test_1';
bag_path = 'C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\MSTAR_BOW\dataSet';

train_data = imageDatastore(train_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
test_data = imageDatastore(test_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
bag_data = imageDatastore(bag_path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
tbl = countEachLabel(train_data)
figure
montage(train_data.Files(1:16:end))
% [trainingSet, validationSet] = splitEachLabel(imds, 0.6, 'randomize');

% extractorFcn = @BagOfFeaturesSURFExtractor;
% bag = bagOfFeatures(bag_data,'CustomExtractor',extractorFcn,'VocabularySize', 5000);
bag = bagOfFeatures(bag_data, 'VocabularySize', 9000);

img = readimage(train_data, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')


categoryClassifier = trainImageCategoryClassifier(train_data, bag);

train_confMatrix = evaluate(categoryClassifier, train_data);
test_confMatrix = evaluate(categoryClassifier, test_data);
mean(diag(test_confMatrix))