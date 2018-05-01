from flask import Flask, render_template, flash, request, make_response
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import os
import gensim, logging
import time
from sklearn.manifold import TSNE
from ggplot import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as col
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# This class displays a form for entering word.
class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
 
 
@app.route("/", methods=['GET', 'POST'])


# This function renders the hello.html template.    
def hello():
    form = ReusableForm(request.form)
    file1 = '' 
    print (form.errors)
    if request.method == 'POST':
        name=request.form['name']
        print (name)
 
        if form.validate():
            # Save the comment here.
            flash('You entered ' + name)
        else:
            flash('All the form fields are required. ')
        
        mymodel = gensim.models.Word2Vec.load('model/mymodel2')
        file1 = simple(model=mymodel, word=name)
    
        flash (mymodel.wv.most_similar(positive=[name], topn=5))
    return render_template('hello.html', form=form, user_image = 'static/' + file1 +'.png')

#@app.route("/images")
# This function generates the t-sne plot for the entered word.
def simple(model, word):
            import io
            from pandas.compat import StringIO
            arr = np.empty((0,100), dtype='f')
            word_labels = [word]

            # get close words
            close_words = model.wv.most_similar(positive=[word], topn=25)
            
            # add the vector for each of the closest words to the array
            arr = np.append(arr, np.array([model[word]]), axis=0)
            for wrd_score in close_words:
                wrd_vector = model[wrd_score[0]]
                word_labels.append(wrd_score[0])
                arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
            # find tsne coords for 3 dimensions
            tsne = TSNE(n_components=3, random_state=0)
            np.set_printoptions(suppress=True)
            Y = tsne.fit_transform(arr)
            
            data = np.random.randint(1, Y.shape[0], Y.shape[0]) 
            
            #%% Create Color Map
            colormap = plt.get_cmap("winter")
            norm = col.Normalize(vmin=min(data), vmax=max(data))


            x_coords = Y[:, 0]
            y_coords = Y[:, 1]
            z_coords = Y[:, 2]
            
            # display scatter plot
            
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(x_coords, y_coords, zs=z_coords, zdir='z', s=20, c=colormap(norm(data)), depthshade=True, marker='o')

            for label, x, y, z in zip(word_labels, x_coords, y_coords, z_coords):
                if label == word:
                    ax.text(x, y, z, '%s' % (label), size=8, zorder=1, color='red')
                else:
                    ax.text(x, y, z, '%s' % (label), size=8, zorder=1, color='k')
            
            # https://datascience.stackexchange.com/questions/17314/are-t-sne-dimensions-meaningful
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_zticks([])
            
            ax.set_xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
            ax.set_ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
            ax.set_zlim(z_coords.min()+0.00005, z_coords.max()+0.00005)
            
            plt.savefig('static/' + word +'.png', bbox_inches='tight')
            
            return word
            
    
if __name__ == "__main__":
    app.run()

