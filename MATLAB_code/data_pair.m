%% Create triplet into a .txt file
bgimg_file_path = '/media/cecilia/DATA/cecilia/labelme/images/bg_file.txt';
triplet_path = '/media/cecilia/DATA/cecilia/labelme/triplet.txt';
mask_path = '/media/cecilia/DATA/cecilia/labelme/masks/';
image_path = '/media/cecilia/DATA/cecilia/labelme/images/';

bgimg_file_fid = fopen(bgimg_file_path);
bgimg_file_struct = cell(ii, 1);
tline = fgets(bgimg_file_fid);
ind = 1;
while ischar(tline)
    bgimg_file_struct{ind} = tline;
    ind  = ind+1;
    tline = fgets(bgimg_file_fid);
end
fclose(bgimg_file_fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%% create triplets
total_bg_img = ind-1;
total_obj = length(obj_Dobj_all);

for k = 1:10
    bg_ind_this = randperm(total_bg_img);
    bg_ind = [bg_ind;bg_ind_this'];
end

obj_ind = randperm(total_obj);

triplet_fid = fopen(triplet_path, 'w');
j = 1;
num_pair = 0;
for i = 1:length(bg_ind)
    % bg image
    this_bg_img = bgimg_file_struct{bg_ind(i), 1};
    this_bg_cellstr = cellstr(this_bg_img);
    
    bg_img = imread(this_bg_cellstr{1,1});
    % object
    obj_this = obj_Dobj_all(obj_ind(j)).annotation;
    this_boundingbox = obj_this.object.boundingbox;
    this_poly_pt = length(obj_this.object.polygon.x');
%    this_area = (this_boundingbox(1,3)-this_boundingbox(1,1)).*(this_boundingbox(1,4)-this_boundingbox(1,2));
    
    while (this_poly_pt <= 6 || this_h >= 0.8 * size(bg_img, 1) || this_w >= 0.8 * size(bg_img, 2))
        j = j + 1;
        j = mod(j, total_obj)+1;
        obj_this = obj_Dobj_all(obj_ind(j)).annotation;
        this_boundingbox = obj_this.object.boundingbox;
        this_poly_pt = length(obj_this.object.polygon.x');
        this_w = (this_boundingbox(1,3)-this_boundingbox(1,1));
        this_h = (this_boundingbox(1,4)-this_boundingbox(1,2));
    end
    
    folder_this = obj_this.folder;
    filename = obj_this.filename;
    mask_file_dir = strcat(mask_path, folder_this, '/', filename);
    img_file_dir = strcat(image_path, folder_this, '/', filename);
    
    % record the bounding box size and the center position
    useful_num = num2str([obj_this.object.boundingbox, obj_this.object.centers]);
    content_cell = strcat(this_bg_img, {' '}, img_file_dir, {' '}, mask_file_dir, {' '}, useful_num);
    fprintf(triplet_fid,'%s\n',content_cell{1,1});
    
    num_pair = num_pair + 1;
    
    j = j + 1;
    j = mod(j, total_obj)+1;
    
end

fclose(triplet_fid);
fprintf('%d pairs have been written to file\n', num_pair)

%% Read background images
bgimg_path = '/media/cecilia/DATA/cecilia/labelme/images/bg_valid.txt';
img_path = '/media/cecilia/DATA/cecilia/labelme/images/';

fid = fopen(bgimg_path);
fid_write = fopen(strcat(img_path, 'bg_file.txt'),'w');

tline = fgets(fid);
ii = 0;
while ischar(tline)
    this_dir = dir(strcat(img_path, tline));
    dirIndex = [this_dir.isdir];  %# Find the index for directories
    fileList = {this_dir(~dirIndex).name}';
    
    for i = 1:length(fileList)
        fprintf(fid_write,'%s\n',strcat(img_path, tline, '/', fileList{i}));
        ii = ii + 1;
    end

    tline = fgets(fid);
end

fprintf('%d files have been written to .txt\n', ii)

fclose(fid);
fclose(fid_write);

%% Save mask image
% for foreground objects
numObj = 5000;
% sorted_count;
% select_name;

% randomly sample 10% of the object
mask_path = '/media/cecilia/DATA/cecilia/labelme_car/masks/';
rand_sample = randi(length(obj_Dobj_all), round(0.1*length(obj_Dobj_all)), 1);

for i = 1:length(obj_Dobj_all)
    
    obj_this = obj_Dobj_all(i).annotation;
    folder_this = obj_this.folder;
    folder_dir = strcat(mask_path, folder_this);
    
    if ~exist(folder_dir, 'dir')
        mkdir(folder_dir)         
    end
    
    filename = obj_this.filename;
    object = obj_this.object;
    
    [mask, class] = LMobjectmask(obj_this, HOMEIMAGES);
    % imshow(colorSegments(mask))
    
    mask_dir = strcat(mask_path, folder_this, '/', filename);
    imwrite(mask, mask_dir);
    
end


%% Pair background and foreground images

% for background images
bg_txt = '/media/cecilia/DATA/cecilia/labelme/images/bg_valid.txt';
folder_path = '/media/cecilia/DATA/cecilia/labelme/images/';
fid = fopen(bg_txt);

tline = fgets(fid);
while ischar(tline)
    
    tline = fgets(fid);
    allFiles = dir(strcat(folder_path,tline));
    allFiles([allFiles.bytes]==0) = [];
    allNames = { allFiles.name };
    disp(tline)
    
end

fclose(fid);