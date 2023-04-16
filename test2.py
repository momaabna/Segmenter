
from segment_anything import sam_model_registry, SamPredictor
import cv2,torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets  import RectangleSelector

#keyboard input library
sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"

last_mask = None
input_point = np.array([[0, 0]])
input_label = np.array([1])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread("examples/10078660_15.tiff")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)



#get mouse click points
def onclick(event):
    global input_point, input_label, last_mask
    x, y = event.xdata, event.ydata
    print(f"Clicked at {x:.0f}, {y:.0f}")
    #check which mouse button was pressed
    if event.button == 2:

        #left click
        #check if ctrl key is pressed
        
        print("Removing point")
        #remove the point
        i = np.argmin(np.sum(np.square(input_point - np.array([[x, y]])), axis=1))
        input_point = np.delete(input_point, i, axis=0)
        input_label = np.delete(input_label, i, axis=0)
    elif event.button == 1:         
        print("Adding positive point")
        input_label = np.concatenate([input_label, [1]], axis=0)
        input_point = np.concatenate([input_point, [[x, y]]], axis=0)

    elif event.button == 3:
        #right click
        print("Adding negative point")
        input_label = np.concatenate([input_label, [0]], axis=0)
        input_point = np.concatenate([input_point, [[x, y]]], axis=0)
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,)
    print(masks.shape, scores.shape, logits.shape)
    #get the best mask

    i = np.argmax(scores)
    mask = masks[i]
    score = scores[i]
    print(f"Best mask score: {score:.3f}")
    #clear the plot
    plt.cla()
    plt.imshow(image)
    show_mask(mask, plt.gca())
    last_mask = mask
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.draw()


plt.figure(figsize=(10,10))
fig,ax = plt.subplots()
ax.imshow(image)
boxes = np.array([[0, 0, 1, 1]])
def line_select_callback(eclick, erelease):
    global boxes,image
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    #if click right button, remove the box
    if eclick.button == 3:
        i = np.argmin(np.sum(np.square(boxes - np.array([[x1, y1, x2, y2]])), axis=1))
        boxes = np.delete(boxes, i, axis=0)
    #new add box to the list
    boxes = np.concatenate([boxes, [[x1, y1, x2, y2]]], axis=0)
    input_boxes = torch.tensor(boxes).reshape(-1, 4).to(device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    plt.cla()
    plt.imshow(image)
    masks = masks.cpu().numpy()
    for mask in masks:
        show_mask(mask, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    # rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), fill=False, edgecolor='red', linewidth=2)
    # ax.add_patch(rect)


rs = RectangleSelector(plt.gca(), line_select_callback)
#plt.imshow(image)
#show_points(input_point, input_label, plt.gca())
#plt.axis('on')
#on key or mouse click
#cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
#save the mask
#cv2.imwrite("mask.png", last_mask*255)


