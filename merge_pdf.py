import glob
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
from fpdf import FPDF
import plotly.graph_objects as go
from IPython.display import Image
import numpy as np
np.random.seed(1)

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
))


fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
))

def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()

    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)

def write_images_to_pdf(output_path, image_list):
    pdf = FPDF()
    # imagelist is the list with all image filenames
    for image in image_list:
        pdf.add_page()
        pdf.image(image)#,5,5,200,150)
    pdf.output(output_path, "F")

if __name__ == '__main__':
    #paths = glob.glob('/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/*.pdf')
    #paths = os.listdir('/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test')
    #paths.sort()
    #merger('/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/pdf_merger.pdf', paths)
    #write_images_to_pdf('/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/pdf_images.pdf', ['/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/img1.png','/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/img2.png'])
    img_bytes1 = fig1.to_image(format="png")
    img_bytes2 = fig2.to_image(format="png")
    write_images_to_pdf('/Users/neelu/StudyProjects/StudyProjects/pdf_merger_test/pdf_images1.pdf',[Image(img_bytes1),Image(img_bytes2)])