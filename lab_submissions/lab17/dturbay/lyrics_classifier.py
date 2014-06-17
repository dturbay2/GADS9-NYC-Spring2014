"""Lyrics classifier for Flask application"""

# Our familiar imports  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# A new import
import pickle

# Local imports for the Million Song Dataset stemming algorithm
from msd.stem import transformLyrics

class LyricsClf():
    """A MultinomialNB classifier for predicting artists from lyrics.
    Offers train, save, and load routines for offline and startup
    purposes. Offers predictArtist for online use.
    """
    def __init__(self,picklefile=None):
        """Constructor that creates an empty artistLabels dictionary,
        a CountVectorizer placeholder, and a classifier placeholder.
        If a picklefile is specified, the returned object is instantiated
        from a pickled version on disk.
        """
        self.artistLabels = dict()
        self.vectorizer = None
        self.clf = None
        if picklefile:
            self.load(picklefile)

    def makeArtistLabels(self,artistList):
        """Creates a mapping between artist names and
        integer class labels.
        """
        for i,artist in enumerate(artistList):
            self.artistLabels[artist] = i 

    def getLabel(self,artist):
        """Returns the integer label for a given artist.
        Returns -1 if an artist does not exist.
        """

        if artist in self.artistList:
            return self.artistLabels[artist]
        else:
            return -1

    def predictArtist(self,lyrics):
        """Returns an artist name given sample song lyrics.
        Applies the Million Song Dataset stemming routine to
        the lyrics (pre-processing), vectorizes the lyrics,
        and runs them through the MultinomialNB classifier.
        Returns the artist name associated with the predicted
        label.
        """
        transformed_lyrics = transformLyrics(lyrics)
        df = pd.DataFrame({'Lyrics':[transformed_lyrics]})
        bow = self.vectorizer.transform(transformed_lyrics)
        y = self.clf.predict(X)

        for artist,label in self.artistLabels.items():
            if label == y:
                return artist
        return "artist not found"

    def train(self,csvfile):
        """Read in a csv of lyrics then do the following:
        - Turn artist names into class labels
        - Build a CountVectorizer
        - Define the classifier's training inputs and outputs
        - Instantiate the classifier
        - Train the classifier
        """
        # Read the input file
        df = pd.read_csv('csvfile')

        # Create a mapping of artist (string) to label (integer)
        artists = df['Artists'].unique()
        self.makeArtistLabels(artists)

        # Create a new column, Label, which will be the model's output label
        df['Label'] = df['Artist'].apply(self.getLabel)
        
        # Create the input and output for training the classifier
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(df['Lyrics'])
        y = df['Label']
        
        # Instantiate and train the classifier
        self.clf = MultinomialNB()
        self.clf.fit(X,y)

    def save(self,picklefile):
        """Save this LyricsClf object to disk as picklefile."""
        pickle.dump(self,open(picklefile,'wb'))

    def load(self,picklefile):
        """Load a LyricsClf object from picklefile.
        Return this loaded object for future use.
        """
        self = pickle.load(open(picklefile,'rb'))
        return self

# end of LyricsClf class
