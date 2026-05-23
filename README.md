# PyTorch Coil Simulator and Optimization
This software grew some struggles with the generally excellent [CoilGen](https://github.com/Philipp-MR/CoilGen) software, which can struggle with situations where you would like to impose custom constraints on the coil geometry. For some problems (like PCB coils), I find that the mesh-based approach offers too much freedom, which can make optimizing coils with many turns painful. 

To address these issues, this project provides a lightweight Bio-Savart simulator that takes a sequence of 3D points that represent a coil. This
simulator is differentiable, making it possible to use PyTorch to optimize coils for whatever objective function you'd like. This project also includes
functions for exporting coil geometry directly to a KiCAD PCB file, since this process was a major sticking point for CoilGen.

## Example: 1D Gradient Optimization with PyTorch
This section describes the process of designing a 1D gradient for a long/skinny ROI. To run the code, see `rect_grad_opt.ipynb` in the `examples` folder.The optimization from CoilGen was producing results that look like this:

![Simulation output from coilgen, showing asymmetric windings](examples/Rectangular%20Gradient/Images/coilgen-output.png)

While the general winding pattern is the same, there are a few issues:
1. The winding pattern should be symmetric for this ROI
2. I need more windings, but CoilGen struggles with more turns and a finer mesh
3. I'd like slightly more control over the connections between turns in order to fit more on the board.
4. For this problem specifically, I would like to force all of the traces to be straight lines to reduce the optimization space.

For this design with the coilsim module, I want to optimize the spacing of the turns only along the gradient direction, while enforcing a 
square spiral winding pattern with a given spacing. The coil will use both sides of a 2 layer PCB, and I will use 2 of these boards for the 
top and bottom of the gradient. The first step is to create a function that maps the parameters to be optimized (the position of each winding) to the 
coil object (which is fundamentally just a list of all of the corner points). To produce a well behaved optimization under gradient descent, instead of taking the coil positions directly, I run the input parameters through `torch.nn.Softplus()`, and normalize the output to get the fractional position of each winding. The resulting coil construction function looks like this:

```
def create_gradcoil(raw_heights, spacing, points1, points2):

    pos_heights = torch.nn.Softplus()(raw_heights)
    heights = (-(torch.cumsum(pos_heights, dim=0))/torch.sum(pos_heights + spacing*turns) * 
                 (length/2 - length/2/(turns) - spacing*turns))
    heights = heights - torch.arange(1, len(heights) + 1) * spacing

    # Build all points as a function of heights
    point_list = []
    for i in range(turns-1, -1, -1):

        s = spacing * i
        h = heights[i]
        
        point_list.append(torch.stack([
            torch.tensor(width/2 - s - spacing/2),
            h,
            torch.tensor(separation/2 + board_thickness/2)
        ]))
        point_list.append(torch.stack([
            torch.tensor(-width/2 + s - spacing/2),
            h,
            torch.tensor(separation/2 + board_thickness/2)
        ]))
        point_list.append(torch.stack([
            torch.tensor(-width/2 + s - spacing/2),
            torch.tensor(-length/2 + s - spacing),
            torch.tensor(separation/2 + board_thickness/2)
        ]))
        point_list.append(torch.stack([
            torch.tensor(width/2 - s + spacing - spacing/2),
            torch.tensor(-length/2 + s - spacing),
            torch.tensor(separation/2 + board_thickness/2)
        ]))

    point_list.append(torch.tensor([width/2 - s + spacing - spacing/2,
                                    0,
                                    separation/2 + board_thickness/2], dtype=torch.float32))


    points = torch.stack(point_list)

    # Flip and negate x,y coordinates (no in-place operations)
    points_flipped = torch.flip(points, dims=[0])
    points_flipped = torch.cat([-points_flipped[:, :2], points_flipped[:, 2:]], dim=1)

    # Concatenate (no in-place operations)
    points_top = torch.cat([points, points_flipped], dim=0)

    points_bottom = torch.flip(points_top, dims=[0])
    points_bottom = torch.cat([points_bottom[1:-1,0:1], -points_bottom[1:-1,1:2], points_bottom[1:-1, 2:] - board_thickness], dim=1)
    points_bottom = torch.cat([points_bottom, torch.stack([
            torch.tensor(-width/2 + spacing * (turns-1 + 1/2)),
            -heights[turns-1],
            torch.tensor(separation/2 - board_thickness/2)
        ]).unsqueeze(0)])

    points1 = torch.cat([points_top, torch.flip(points_bottom, dims=[0])])

    # Create mirrored coil (no in-place operations)
    points2 = torch.cat([points1[:, :2], points1[:, 2:] - separation], dim=1)
    
    # Create Coil objects
    gc1 = coilsim.Coil(points1)
    gc2 = coilsim.Coil(points2)

    return gc1, gc2
```
The initial state uses even spacing between the windings, yielding a coil that looks like this:

![The initial state of the coil, showing the field pattern as different colors on the surface of the ROI](examples/Rectangular%20Gradient/Images/optimization_initial_state.png)

The cylindrical ROI is colored with the flux density at each point within it. The optimization follows a pretty typical PyTorch setup:
```
with tqdm(range(num_epochs)) as titer:
    for i in titer:
        optimizer.zero_grad()
        gc1, gc2 = create_gradcoil(raw_heights,spacing,points_combined, points_sc)
        Bz = gc1.biot_savart(observation_points, 1)[:,2] + gc2.biot_savart(observation_points, 1)[:,2]
        # print(heights)
        bz_norm = Bz/torch.max(Bz)
        y = y_norm
        N = y.shape[0]
        slope = (N * torch.sum(y * Bz) - torch.sum(y) * torch.sum(Bz)) / (N * torch.sum(y**2) - torch.sum(y)**2)
        # print(Bz)
        slopes.append(slope.detach())
        loss = torch.sum((y_norm - bz_norm)**2)# - torch.sigmoid(slope*1000)
        # loss = torch.sum((y_norm + bz_norm)**2)
        # print(loss)z
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        titer.set_postfix(loss=loss.item())

print(f'Gradient Strength: {slopes[-1] * 20500}mT/m')
```

Here, different loss functions yield subtly different results. Optimizing for both linearity and gradient strength unsurprisingly results in a less uniform field strength than optimizing for uniformity alone.

After optimization, the coil looks like this:

![The final state of the coil, showing the field pattern as different colors on the surface of the ROI](examples/Rectangular%20Gradient/Images/optimization_final_state.png).

Plotting the field value along the axis of the ROI gives a better picture of the optimization:

![Field maps for before and after optimization, showing better linearity afte optimization](examples/Rectangular%20Gradient/Images/fieldmap.png).

Finally, this can be converted to a KiCAD PCB using this snippet:
```
def write_coil(pcb, startline, coil, width, net, center):
    boardlines = []
    with open(pcb, 'r') as board:
        boardlines = board.readlines()

    with open(pcb, 'w') as board:
        for i in range(startline-1):
            board.write(boardlines[i])

        endline = startline
        for segment in coil.get_segments().detach().numpy():
            if(segment[0,2] != segment[1,2]): #Via, don't draw
                pass
            layer = "F.Cu" if segment[0,2] > np.mean(gc1.points.detach().numpy()[:,2]) else "B.Cu"
            board.write(segment_string(segment[0,0:2]*1000 + center, 
                                       segment[1,0:2]*1000 + center,
                                       width,layer,net))
            endline += 8

        for i in range(startline-1, len(boardlines)):
            board.write(boardlines[i])

    mdfile = pcb.split('.')[0] + '.yaml'
    with open(mdfile, 'w') as file:
        file.write(f'start_line: {startline}\n')
        file.write(f'end_line: {endline}')

write_coil(pcbfile, 307, gc3, 0.8, 1, np.array([100,100]))
```

![The finished gradient coil after manufacturing](examples/Rectangular%20Gradient/Images/gradpcb.jpg)