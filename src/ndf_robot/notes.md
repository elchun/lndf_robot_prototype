# QUESTIONS FOR 2022 Jan 18

- Why is file laid out this way (i.e. can i refactor to be more
function based)

# FUTURE IMPROVEMENTS

- Use occupancy of real gripper but any query points

# Meeting 2022-1-19

Main goal: get next project direction

Slack message: 

Hey Yilun,
I’ve been working on integrating the occupancy biasing from ndf_demo with evaluate_ndf.py and I’ve noticed a few things.

I think the current evaluate uses query points without the 0.105 shift so that the query points actually intersect the mug.  This seems to actually produce better performance, at least when using the full gripper shape (I ran evaluate with the full gripper shifted and not shifted and shifted has very low (< 0.33) place success).  The consequence tho is that using occupancy is a little more challenging.  I think it can be done by checking occupancy of a separate point cloud that is shifted relative to the original and uses the full gripper shape, but it will likely take me a little longer to implement.

Use different poitn cloud

I’m not actually sure how much using the occupancy can help since it seems that solutions fail for other reasons (at least on ndf_demo, the common failure mode is the gripper is nowhere near where it is supposed to be)

run optimizer longer

Some of the point clouds from the demo depth cameras look a little messed up (see screenshots).  Is this normal?

- from fusing depth camera images, real world artifact

- joint loss

At any rate, do you think we could meet tomorrow or Wednesday to check in on the project and next steps?

Put bias outside, 

Understand how we go from poses to trajectory

ping prof kaelbling on csail account
