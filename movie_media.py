import webbrowser
import fresh_tomatoes

class Movie():
    '''
    Defines a Movie
    '''
    valid_ratings = ["G","R","PG","PG-13"]
    def __init__(self,title,storyline,trailer_link,poster_img,rating):
        self.title = title
        self.trailer = trailer_link
        self.storyline = storyline
        self.poster = poster_img
        self.rating = rating

    def show_trailor(self):
        webbrowser.open(self.trailer)

if __name__ == '__main__':
    raazi = Movie("Raazi","Story of a Girl defending her country","https://www.youtube.com/watch?v=nDbjJVmGV98","https://akm-img-a-in.tosshub.com/indiatoday/images/story/201805/raazi_alia_bhatt.jpeg?24QuIeLVGvVLuGmwb6vKwLwmvXpfNwWK","PG-13")
    #raazi.show_trailor()

    movie_list = [raazi]
    fresh_tomatoes.open_movies_page(movie_list)

    print(Movie.__doc__)