load('splits.mat', 'trainNdxs', 'testNdxs');
load('nyu_depth_v2_labeled.mat', 'scenes');

train_set = unique(scenes(trainNdxs));
test_set = unique(scenes(testNdxs));

f = fopen('nyu_train.txt', 'w');
for i = 1:size(train_set)
    fprintf(f, '%s\n', train_set{i});
end
fclose(f);

f = fopen('nyu_test.txt', 'w');
for i = 1:size(test_set)
    fprintf(f, '%s\n', test_set{i});
end
fclose(f);
