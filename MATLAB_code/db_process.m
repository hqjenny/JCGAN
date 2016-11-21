HOMEANNOTATIONS = '/media/cecilia/DATA/cecilia/labelme/annotation';
HOMEIMAGES = '/media/cecilia/DATA/cecilia/labelme/images';
D = LMdatabase(HOMEANNOTATIONS);

NEWHOMEANNOTATIONS = '/media/cecilia/DATA/cecilia/labelme_car/annotations';
NEWHOMEIMAGES = '/media/cecilia/DATA/cecilia/labelme_car/images';

% [D_occlude_people, ~] = LMquery(D_occlude, 'object.name', 'person');
% LMdbshowscenes(D_occlude_people, HOMEIMAGES);
% 
% [D_occlude_tree, ~] = LMquery(D_occlude, 'object.name', 'tree');
% LMdbshowscenes(D_occlude_tree, HOMEIMAGES);
[~, name, ~] = fileparts(D_occlude_tree(2).annotation.filename);
name = strcat(name, '.xml');
filename = fullfile(HOMEANNOTATIONS, D_occlude_tree(2).annotation.folder, name);
[annotation, img] = LMread(filename, HOMEIMAGES);
LMplot(annotation, img)
[newannotation, newimg, crop, scaling, err, msg] = ...
    LMcookimage(annotation, img, 'objectname', 'car', 'objectlocation', 'centered', 'maximagesize', [256 256], 'impad', 255);
LMplot(newannotation, newimg)

%% ALL CAR OBJECTS
NEWHOMEIMAGES = '/media/cecilia/DATA/cecilia/labelme_car_all/images';
NEWHOMEANNOTATIONS = '/media/cecilia/DATA/cecilia/labelme_car_all/annotations';
if exist(NEWHOMEIMAGES, 'dir')
    rmdir(NEWHOMEIMAGES)
end
if exist(NEWHOMEANNOTATIONS, 'dir')
    rmdir(NEWHOMEANNOTATIONS)
end
[D_car, ~] = LMquery(D, 'object.name', 'car-occluded');
LMcookdatabase(D_car, HOMEIMAGES, HOMEANNOTATIONS, NEWHOMEIMAGES, NEWHOMEANNOTATIONS, ...
    'minobjectsize', [64 64], 'maximagesize', [256, 256])
D_car_all = LMdatabase(NEWHOMEANNOTATIONS);

%% SAVE IMAGE PAIR SCALE 256 * 256
[D_occlude,j_occlude] = LMquery(D, 'object.occluded', 'no');
[D_occlude_car, ~] = LMquery(D_occlude, 'object.name', 'car');
% display all qualified images
LMdbshowscenes(D_occlude_car, HOMEIMAGES);
if exist(NEWHOMEIMAGES, 'dir')
    rmdir(NEWHOMEIMAGES)
end
if exist(NEWHOMEANNOTATIONS, 'dir')
    rmdir(NEWHOMEANNOTATIONS)
end
LMcookdatabase(D_occlude_car, HOMEIMAGES, HOMEANNOTATIONS, NEWHOMEIMAGES, NEWHOMEANNOTATIONS, ...
    'minobjectsize', [64 64], 'maximagesize', [256, 256])
D_car = LMdatabase(NEWHOMEANNOTATIONS);

% save mask
mask_path = '/media/cecilia/DATA/cecilia/labelme_car_all/masks_filter/';
img_path = '/media/cecilia/DATA/cecilia/labelme_car_all/images_filter/';
if ~exist(img_path,'dir')
    mkdir (img_path)
end
if ~exist(mask_path,'dir')
    mkdir (mask_path)
end
save_pair(img_path, mask_path, D_car_all, NEWHOMEIMAGES);

%% CONVERT ALL REAL IMAGES TO 256 * 256, PNG FORMAT
txt_path = '/home/JC/JCGAN/real.txt';
png_path = '/home/JC/JCGAN/real_png.txt'; % Duplicated txt file of real.txt
new_real_img_path = '/media/cecilia/DATA/cecilia/labelme_png/images';
if ~exist(new_real_img_path, 'dir')
    mkdir(new_real_img_path)
end

real_fid = fopen(txt_path);
real_png_fid = fopen(png_path);
tline = fgets(real_fid);
tline1 = fgets(real_png_fid);
ind = 0;
while ischar(tline)
    
    tline = cellstr(tline);
    tline1 = cellstr(tline1);

    ind  = ind+1;
    
    img = imread(tline{1});
    
    [~, newimg, ~, ~, ~, ~] = LMcookimage(annotation, img,...
       'objectlocation', 'original', 'maximagesize', [256 256], 'impad', 0);
    
    [pathstr,name,~] = fileparts(tline1{1});
    folder_name = pathstr;
    if ~exist(folder_name, 'dir')
        mkdir(folder_name)
    end
    file_path = strcat(pathstr, '/', name);
    imwrite(newimg, file_path, 'PNG')
    
    tline = fgets(real_fid);
    tline1 = fgets(real_png_fid);
    
    fprintf( 'converting %s \n', file_path )

end
fprintf('finish converting %d images to png format\n', ind)
fclose(real_fid);
