

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
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
#on key or mouse click
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
#save the mask
cv2.imwrite("mask.png", last_mask*255)


