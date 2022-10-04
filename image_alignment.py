#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    demo.py [options] <yaml-config> <checkpoint> <images>...
    demo.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <images>                      Path to images

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
"""

#python image_alignment.py -d 0 config/wireframe.yaml trained_models/pretrained.pth.tar "data/unimelb_corridor/pairs_clear_notexture/r001.png" "data/unimelb_corridor/pairs_clear_notexture/s001.png" 

from calendar import different_locale
from operator import index
import os
import os.path as osp
import pprint
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

import math

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    checkpoint = torch.load(args["<checkpoint>"], map_location=device)

    # Load model
    model = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_index = 0

    for imname in args["<images>"]:
        print(f"Processing {imname}")
        im = skimage.io.imread(imname)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        if (image_index==0):
            imname1 = imname[imname.rfind('/')+1:-4]
            dir_path = os.path.join(os.path.dirname(imname), f"{imname1}_output")
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            im1 = im
        else:
            imname2 = imname[imname.rfind('/')+1:-4]
            im2 = im

        def angle_LoC (P1, P2, P3):
            a = np.linalg.norm(P2 - P1)
            b = np.linalg.norm(P3 - P1)
            c = np.linalg.norm(P3 - P2)
            return math.acos((a**2 + b**2 - c**2)/(2*a*b))

        def directional_angle(A, B):
            y_a = A[0]
            x_a = A[1]
            y_b = B[0]
            x_b = B[1]

            print(f"A: {A}")
            print(f"B: {B}")

            delta_y = y_b - y_a
            delta_x = x_b - x_a

            print(f"delta_y: {delta_y}")
            print(f"delta_x: {delta_x}")

            if (delta_x < 0):
                return math.atan(delta_y/delta_x) + math.pi
            elif (delta_y < 0):
                return math.atan(delta_y/delta_x) + 2*math.pi
            else:
                return math.atan(delta_y/delta_x)


        def angle_from_directionals(P1, P2,P3):
            ni_P1_P2 = directional_angle(P1, P2)
            ni_P1_P3 = directional_angle(P1, P3)
            print(f"ni_P1_P2: {ni_P1_P2}")
            print(f"ni_P1_P3: {ni_P1_P3}")

            if (ni_P1_P3 > ni_P1_P2):
                return (ni_P1_P3 - ni_P1_P2)
            else:
                return (2*math.pi - (ni_P1_P2 - ni_P1_P3))

        #parameters
        if (image_index == 0):
            thr_conf = 0.98
            thr_conf_rgb = thr_conf
        else:
            thr_conf = 0.98
            thr_conf_synt = thr_conf

        eucl_thr = 0.10

        thr_dist = 1

        normalisation = True

        #n = np.sum(nscores > thr_conf)

        pointsA = np.empty((0, 2))
        pointsB = np.empty((0, 2))

        j = 0
        for i in range(0, len(nscores)):
            if nscores[i] > thr_conf:

                print(np.linalg.norm(nlines[i, 0] - nlines[i, 1]))
                
                if (np.linalg.norm(nlines[i, 0] - nlines[i, 1]) > 3):
                    pointsA = np.append(pointsA, np.expand_dims(nlines[i, 0], axis = 0), axis = 0)
                    pointsB = np.append(pointsB, np.expand_dims(nlines[i, 1], axis = 0), axis = 0)
                    #pointsA[j] = nlines[i, 0]
                    #pointsB[j] = nlines[i, 1]
                    j += 1

                    print(f"Line {j}, nscore {nscores[i]} \nPoint A: {nlines[i, 0]}, Point B: {nlines[i, 1]}")

        middle = np.empty((j))

        tri_descriptor = np.empty((0, 4))
        tri_lines = np.empty((0, 4, 2))

        pointK = np.empty((2))
        pointL = np.empty((2))

        for i in range(0, len(pointsA)):
            tiesK_A = np.empty((0,2))
            tiesK_B = np.empty((0,2))
            tiesL_A = np.empty((0,2))
            tiesL_B = np.empty((0,2))

            pointK = pointsA[i]
            pointL = pointsB[i]

            for j in range(0, len(pointsA)):
                if i != j:
                    if np.linalg.norm(pointK - pointsA[j]) < thr_dist:
                        tiesK_A = np.append(tiesK_A, np.expand_dims(pointsA[j], axis = 0), axis = 0)
                        tiesK_B = np.append(tiesK_B, np.expand_dims(pointsB[j], axis = 0), axis = 0)
                    if np.linalg.norm(pointK - pointsB[j]) < thr_dist:
                        tiesK_A = np.append(tiesK_A, np.expand_dims(pointsB[j], axis = 0), axis = 0)
                        tiesK_B = np.append(tiesK_B, np.expand_dims(pointsA[j], axis = 0), axis = 0)
                    if np.linalg.norm(pointL - pointsA[j]) < thr_dist:
                        tiesL_A = np.append(tiesL_A, np.expand_dims(pointsA[j], axis = 0), axis = 0)
                        tiesL_B = np.append(tiesL_B, np.expand_dims(pointsB[j], axis = 0), axis = 0)
                    if np.linalg.norm(pointL - pointsB[j]) < thr_dist:
                        tiesL_A = np.append(tiesL_A, np.expand_dims(pointsB[j], axis = 0), axis = 0)
                        tiesL_B = np.append(tiesL_B, np.expand_dims(pointsA[j], axis = 0), axis = 0)
            if (len(tiesK_A) != 0 and len(tiesL_A) != 0):
                middle[i] = True

                for m in range(0, len(tiesK_A)):
                    print(f"m: {m}")
                    #d1 = angle_LoC(tiesK_A[m], tiesL_A[0], tiesK_B[m])
                    d1 = angle_from_directionals(tiesK_A[m], tiesL_A[0], tiesK_B[m])
                    print(f"d1: {d1}")

                    d3 = np.linalg.norm(tiesK_A[m] - tiesL_A[0])/np.linalg.norm(tiesK_A[m] - tiesK_B[m])

                    for n in range(0, len(tiesL_A)):
                        print(f"n: {n}")
                        #d2 = angle_LoC(tiesL_A[n], tiesK_A[0], tiesL_B[n])
                        d2 = angle_from_directionals(tiesL_A[n], tiesK_A[0], tiesL_B[n])
                        print(f"d2: {d2}")

                        d4 = np.linalg.norm(tiesL_A[n] - tiesK_A[0])/np.linalg.norm(tiesL_A[n] - tiesL_B[n])

                        tri_descriptor = np.append(tri_descriptor, [[d1, d2, d3, d4]], axis = 0)

                        tri_lines = np.append(tri_lines, [[tiesK_B[m], tiesK_A[m], tiesL_A[n], tiesL_B[n]]], axis = 0)
            else:
                middle[i] = False

            # if i == 200:
            #     print("tiesK_A:")
            #     print(tiesK_A)
            #     print("tiesK_B:")
            #     print(tiesK_B)
            #     print("tiesL_A:")
            #     print(tiesL_A)
            #     print("tiesL_B:")
            #     print(tiesL_B)

        print(middle)

        print(f"Number of descriptors created: {len(tri_lines)}")

        if (image_index == 0):
            #np.savetxt(os.path.join(dir_path, f"tri_descriptors_{imname1}.txt"), tri_descriptor, delimiter = ' ')
            #np.savetxt(os.path.join(dir_path, f"tri_lines_{imname1}.txt"), tri_lines, delimiter = ' ')
            tri_descriptor_one = tri_descriptor
            tri_lines_1 = tri_lines

        else:
            #np.savetxt(os.path.join(dir_path, f"tri_descriptors_{imname2}.txt"), tri_descriptor, delimiter = ' ')
            #np.savetxt(os.path.join(dir_path, f"tri_lines_{imname2}.txt"), tri_lines, delimiter = ' ')
            tri_descriptor_two = tri_descriptor
            tri_lines_2 = tri_lines

        #Plot all the lines with confidence above a certain threshold, with the middle segments of trilines on top.
        #Plots two subplots for two images (real one and synthetic one) using the image_index variable
        if (image_index == 0):
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)

        for (a, b), s in zip(nlines, nscores):
            if s < thr_conf:
                continue
            axs[image_index].plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            axs[image_index].scatter(a[1], a[0], **PLTOPTS)
            axs[image_index].scatter(b[1], b[0], **PLTOPTS)
        for a, b, t in zip(pointsA, pointsB, middle):
            if t == True:
                axs[image_index].plot([a[1], b[1]], [a[0], b[0]], c='blue', linewidth=2, zorder=4)
                axs[image_index].scatter(a[1], a[0], **PLTOPTS)
                axs[image_index].scatter(b[1], b[0], **PLTOPTS)
        axs[image_index].imshow(im)
        axs[image_index].axis('off')

        if (image_index==1):
            plt.savefig(os.path.join(dir_path, f"line_segments_{imname1}_{imname2}_{thr_conf_rgb}_{thr_conf_synt}_{eucl_thr}.png"), bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

        image_index = image_index + 1

    #tri_descriptor_one = np.loadtxt(os.path.join(dir_path, f"tri_descriptors_{imname1}.txt"), delimiter = ' ')
    #tri_descriptor_two = np.loadtxt(os.path.join(dir_path, f"tri_descriptors_{imname1}.txt"), delimiter = ' ')

    if normalisation:
        d3_d4_max = np.maximum(tri_descriptor_one[:, [2, 3]].max(), tri_descriptor_two[:, [2, 3]].max())
        tri_descriptor_one = tri_descriptor_one / np.array([2*math.pi, 2*math.pi, d3_d4_max, d3_d4_max])
        tri_descriptor_two = tri_descriptor_two / np.array([2*math.pi, 2*math.pi, d3_d4_max, d3_d4_max])

    descriptor_match_indices = np.empty((0,3))

    n_matches = 0

    for i in range(0, len(tri_descriptor_one)):
        for j in range(0, len(tri_descriptor_two)):
            d = np.linalg.norm(tri_descriptor_one[i] - tri_descriptor_two[j])
            if d < eucl_thr:
                n_matches += 1
                descriptor_match_indices = np.append(descriptor_match_indices, [[i, j, d]], axis = 0)
                print(f"Descriptor match! RGB {i+1} and synthetic {j+1} with distance of {d}")

    
    print(f"Total number of descriptor matches: {n_matches}\n")

    if (n_matches != 0):
        print("Descriptor matches:")
        print(descriptor_match_indices + [1, 1, 0])

        matched_descriptors = np.empty((0,5))

        matches_path = os.path.join(dir_path, f"tri_line_matches_{imname1}_{imname2}_{thr_conf_rgb}_{thr_conf_synt}_{eucl_thr}")
        if not os.path.exists(matches_path):
            os.mkdir(matches_path)

        for k in range(0, n_matches, 1):
            fig_m, ax = plt.subplots(1, 2)
            fig_m.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)

            (in1, in2, e_d) = descriptor_match_indices[k]
            in1 = int(in1)
            in2 = int(in2)

            (po1, po2, po3, po4) = tri_lines_1[in1]

            ax[0].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
            ax[0].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
            ax[0].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
            ax[0].scatter(po1[1], po1[0], **PLTOPTS)
            ax[0].scatter(po2[1], po2[0], **PLTOPTS)
            ax[0].scatter(po3[1], po3[0], **PLTOPTS)
            ax[0].scatter(po4[1], po4[0], **PLTOPTS)

            ax[0].imshow(im1)
            ax[0].axis('off')
            ax[0].set_title(f"RGB triline {in1 + 1}")

            (po1, po2, po3, po4) = tri_lines_2[in2]

            ax[1].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
            ax[1].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
            ax[1].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
            ax[1].scatter(po1[1], po1[0], **PLTOPTS)
            ax[1].scatter(po2[1], po2[0], **PLTOPTS)
            ax[1].scatter(po3[1], po3[0], **PLTOPTS)
            ax[1].scatter(po4[1], po4[0], **PLTOPTS)

            ax[1].imshow(im2)
            ax[1].axis('off')
            ax[1].set_title(f"Synth triline {in2 + 1}")

            matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in1+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_one[in1], axis = 0), axis = 1), axis = 0)
            matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in2+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_two[in2], axis = 0), axis = 1), axis = 0)
            matched_descriptors = np.append(matched_descriptors, [[e_d, 0, 0, 0, 0]], axis = 0)

            plt.savefig(os.path.join(matches_path, f"RGB_triline_{in1+1}_Synthetic_triline_{in2+1}.png"), bbox_inches='tight', dpi=300)
            #plt.subplot_tool()
            #plt.tight_layout
            plt.show()
            plt.close()
        
        indices1 = np.arange(1, len(tri_descriptor_one) + 1, 1)
        indices2 = np.arange(1, len(tri_descriptor_two) + 1, 1)

        with open(os.path.join(dir_path, f"output_{imname1}_{imname2}_{thr_conf_rgb}_{thr_conf_synt}_{eucl_thr}.txt"), "w") as f:
            f.write(f"Output of matching of images: RGB {imname1} and {imname2}\n")
            f.write("\nParameters:\n")
            f.write(f"\nSensitivity RGB: {thr_conf_rgb}\n")
            f.write(f"Sensitivity synthetic: {thr_conf_synt}\n")
            f.write(f"Distance between points threshold: {thr_dist} pixels\n")
            f.write(f"Euclidean distance for a match: {eucl_thr}\n")
            if normalisation:
                f.write("Normalisation: Yes\n")
            else:
                f.write("Normalisation: No\n")
            f.write(f"\nExtracted descriptors of RGB {imname1}:\n")
            np.savetxt(f, np.append(np.transpose(np.expand_dims(indices1, axis = 0)), tri_descriptor_one, axis = 1), delimiter = ' ', fmt='%1.3f')
            f.write(f"\nExtracted descriptors of Synthetic {imname2}:\n")
            np.savetxt(f, np.append(np.transpose(np.expand_dims(indices2, axis = 0)), tri_descriptor_two, axis = 1), delimiter = ' ', fmt='%1.3f')
            f.write(f"\nMatched descritpor indices:\n")
            np.savetxt(f, descriptor_match_indices[:, [0, 1]] + 1, delimiter = ' ', fmt='%i')
            f.write(f"\nMatched descriptors:\n")
            np.savetxt(f, matched_descriptors, delimiter = ' ', fmt='%1.3f')
    else:
        print("No matches found!")

        indices1 = np.arange(1, len(tri_descriptor_one) + 1, 1)
        indices2 = np.arange(1, len(tri_descriptor_two) + 1, 1)

        with open(os.path.join(dir_path, f"output_{imname1}_{imname2}_{thr_conf_rgb}_{thr_conf_synt}_{eucl_thr}.txt"), "w") as f:
            f.write(f"Output of matching of images: RGB {imname1} and {imname2}\n")
            f.write("\nParameters:\n")
            f.write(f"\nSensitivity RGB: {thr_conf_rgb}\n")
            f.write(f"Sensitivity synthetic: {thr_conf_synt}\n")
            f.write(f"Distance between points threshold: {thr_dist} pixels\n")
            f.write(f"Euclidean distance for a match: {eucl_thr}\n")
            if normalisation:
                f.write("Normalisation: Yes\n")
            else:
                f.write("Normalisation: No\n")
            f.write(f"\nExtracted descriptors of RGB {imname1}:\n")
            np.savetxt(f, np.append(np.transpose(np.expand_dims(indices1, axis = 0)), tri_descriptor_one, axis = 1), delimiter = ' ', fmt='%1.3f')
            f.write(f"\nExtracted descriptors of Synthetic {imname2}:\n")
            np.savetxt(f, np.append(np.transpose(np.expand_dims(indices2, axis = 0)), tri_descriptor_two, axis = 1), delimiter = ' ', fmt='%1.3f')

if __name__ == "__main__":
    main()




# if (n_matches == 1):
#             (in1, in2, e_d) = descriptor_match_indices[0]
#             in1 = int(in1)
#             in2 = int(in2)

#             (po1, po2, po3, po4) = tri_lines_1[in1]

#             ax[0].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
#             ax[0].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
#             ax[0].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
#             ax[0].scatter(po1[1], po1[0], **PLTOPTS)
#             ax[0].scatter(po2[1], po2[0], **PLTOPTS)
#             ax[0].scatter(po3[1], po3[0], **PLTOPTS)
#             ax[0].scatter(po4[1], po4[0], **PLTOPTS)

#             ax[0].imshow(im1)
#             ax[0].axis('off')
#             ax[0].set_title(f"RGB {imname1}, triline {in1 + 1}")

#             (po1, po2, po3, po4) = tri_lines_2[in2]

#             ax[1].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
#             ax[1].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
#             ax[1].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
#             ax[1].scatter(po1[1], po1[0], **PLTOPTS)
#             ax[1].scatter(po2[1], po2[0], **PLTOPTS)
#             ax[1].scatter(po3[1], po3[0], **PLTOPTS)
#             ax[1].scatter(po4[1], po4[0], **PLTOPTS)

#             ax[1].imshow(im2)
#             ax[1].axis('off')
#             ax[1].set_title(f"Synth {imname2}, triline {in2 + 1}")

#             matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in1+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_one[in1], axis = 0), axis = 1), axis = 0)
#             matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in2+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_two[in2], axis = 0), axis = 1), axis = 0)
#             matched_descriptors = np.append(matched_descriptors, [[e_d, 0, 0, 0, 0]], axis = 0)
#         else:
#             for k in range(0, n_matches, 1):

#                 (in1, in2, e_d) = descriptor_match_indices[k]
#                 in1 = int(in1)
#                 in2 = int(in2)

#                 (po1, po2, po3, po4) = tri_lines_1[in1]

#                 ax[k, 0].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 0].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 0].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 0].scatter(po1[1], po1[0], **PLTOPTS)
#                 ax[k, 0].scatter(po2[1], po2[0], **PLTOPTS)
#                 ax[k, 0].scatter(po3[1], po3[0], **PLTOPTS)
#                 ax[k, 0].scatter(po4[1], po4[0], **PLTOPTS)

#                 ax[k, 0].imshow(im1)
#                 ax[k, 0].axis('off')
#                 ax[k, 0].set_title(f"RGB {imname1}, triline {in1 + 1}")

#                 (po1, po2, po3, po4) = tri_lines_2[in2]

#                 ax[k, 1].plot([po1[1], po2[1]], [po1[0], po2[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 1].plot([po2[1], po3[1]], [po2[0], po3[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 1].plot([po3[1], po4[1]], [po3[0], po4[0]], c='green', linewidth=2, zorder=4)
#                 ax[k, 1].scatter(po1[1], po1[0], **PLTOPTS)
#                 ax[k, 1].scatter(po2[1], po2[0], **PLTOPTS)
#                 ax[k, 1].scatter(po3[1], po3[0], **PLTOPTS)
#                 ax[k, 1].scatter(po4[1], po4[0], **PLTOPTS)

#                 ax[k, 1].imshow(im2)
#                 ax[k, 1].axis('off')
#                 ax[k, 1].set_title(f"Synth {imname2}, triline {in2 + 1}")

#                 matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in1+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_one[in1], axis = 0), axis = 1), axis = 0)
#                 matched_descriptors = np.append(matched_descriptors, np.append(np.expand_dims(np.expand_dims(in2+1, axis = 0), axis = 1), np.expand_dims(tri_descriptor_two[in2], axis = 0), axis = 1), axis = 0)
#                 matched_descriptors = np.append(matched_descriptors, [[e_d, 0, 0, 0, 0]], axis = 0)