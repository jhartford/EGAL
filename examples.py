import numpy as np
from transformers import (
    BertModel,
    BertTokenizer
)
import pickle

def text_to_embedding_fn(text, tokenizer, model):
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    outputs = model(input_ids)
    return outputs

examples = {'drill': {0:"And Coach Mackie does every drill and exercise with his players, not because he has to, but because that's the way he has always done it.",
           1:"Use an electric drill to punch holes of various sizes into the wood."
           },
            "back":{0:"He had huge shoulders and a broad back which tapered to an extraordinarily small waist.",
                    1:"Marcia is not interested in getting her job back, but wishes to warn others."},
            "card":{0:"However, if you use your card for a cash withdrawal you will be charged interest from day one.",
                    1:"Your computer audio card has great sound, so what really matters are your PC's speakers."
            },
            "character":{0:"Each character has an interesting back story which explains how they got to be on the plane.",
            1:"The latest version of Microsoft's tablet PC software is capable of recognising even an ill-formed character written on the screen using the pen."},
            "cooler":{0:"Be happy that you found someone cooler to hang out with.",
                      1:"The rasperry pi ran far more smoothly after i added a cooler to it."},
            "fans":{0:"Also 120 mm fans will be a lot quieter then an 80 mm fan moving the same amount of air.",
                    1:"More than in other sports, football fans recollect jerseys and numbers as opposed to faces."},
            "ford":{0:"Rob Ford announced last month that he would not seek re-election",
                    1:"This year Ford released an exciting lineup of SUVs, crossovers, hybrids, trucks and vans"},
            "hard":{0:"That brings me to my next point: never ever attempt to write to a hard drive that is corrupt.",
                    1:"It's always hard to predict exactly what the challenges will be."},
            "house":{0:"Inside, the house is on three storeys, with the ground floor including a drawing room, study and dining room.",
                    1:"After Robert's Rebellion, House Baratheon split into three branches: Lord Robert Baratheon was crowned king and took residence at King's Landing"},
            "jobs":{0:"In Kabul, they usually have low-paying, menial jobs such as janitorial work.",
                    1:"Steve Jobs demonstrating the iPhone 4 to Russian President Dmitry Medvedev"},
            "manual":{0: "Power is handled by a five-speed manual gearbox that is beefed up, along with the clutch - make that a very heavy clutch.",
                      1: "What is going through my head, is that this guy is reading the instructions directly from the manual, which I can now recite by rote."},
            "memory":{1:"Their memory for both items and the associated remember or forget cues was then tested with recall and recognition.",
                    0:"Thanks to virtual memory technology, software can use more memory than is physically present."},
            "ride":{0:"I learned to play on a kit with a hi-hat, a crash cymbal, and a ride cymbal.",
                    1:"It is, for example, a great deal easier to demonstrate how to ride a bicycle than to verbalize it."},
            "stick":{0:"Oskar, disgusted that the singing children are so undisciplined, pulls out his stick and begins to drum.",
                    1:"Though the majority of players use a one-piece stick, the curve of the blade still often requires work."},
            "tank":{0:"There are temperature controlled fermenters and a storage tank, and good new oak barrels for maturation.",
                    1:"First, the front warhead destroys the reactive armour and then the rear warhead has free passage to penetrate the main body armour of the tank."},
            "video":{0:"Besides, how can you resist a band that makes a video in which they rock their guts out while naked and flat on their backs?",
                    1:"You do need to make certain that your system is equipped with a good-quality video card and a sound card."},
            "body":{0:"The human body has a skeleton of bones and our cells have a framework made of a filamentous network.",
                    1:"The designs in question were designs for spare parts for cars, including body panels."},
            "case":{0:"It comes with a protective carrying case and software.",
                    1:"The cost of bringing the case to court meant the amount he owed had risen to £962.50."},
            "club":{0:"Although he has played some club matches, this will be his initial first-class game.",
                    1:"The key to good tempo is to keep the club speed the same during the backswing and the downswing."},
            "drive":{1:"If we're out in the car, lost in an area we've never visited before, he would rather we drive round aimlessly for hours than ask directions.",
                    0:"This means you can record to the hard drive for temporary storage or to DVDs for archiving."},
            "fit":{1:"The trousers were a little long in the leg but other than that the clothes fit fine.",
                   0:"The only way to get fit is to make exercise a regularly scheduled part of every week, even every day."},
            "goals":{0:"For the record, the Brazilian Ronaldo scored two goals in that World Cup final win two years ago.",
                    1:"As Africa attempts to achieve ambitious millennium development goals, many critical challenges confront healthcare systems."},
            "hero":{1:"As a nation, we ought to be thankful for the courage of this unsung hero who sacrificed much to protect society.",
                    0:"Axe is a more useful hero than the Night Stalker"},
            "interest":{1:"The group gig together about four times a week and have attracted considerable interest from record companies.",
                        0:"The bank will not lend money, and interest payments and receipts are forbidden."},
            "magic":{0:"Commonly, sorcerers might carry a magic implement to store power in, so the recitation of a whole spell wouldn't be necessary.",
                    1:"Earvin Magic Johnson dominated the court as one of the world's best basketball players for more than a decade."},
            "market":{0:"It's very well established that the U.S. stock market often leads foreign markets.",
                      1:"I have seen dandelion leaves on sale in a French market and they make a tasty addition to salads - again they have to be young and tender."},
            "plane":{1:"Despite the gusting winds and hazardous visibility, the plane and helicopters kept flying.",
                    0:"Use a scraping plane or rasp on cut edges to smooth any roughness."},
            "router":{0:"But shapers are expensive and a router lacks the power to cut the profile in one pass.",
                      1:"Testing with a router and a notebook computer in the empty townhouse before we moved in, the signal looked good initially."},
            "stream":{0:"The stream and river offers excellent fly fishing for trout and bass; hook a smallmouth on a six weight rod and you're having fun.",
                      1:"You can either listen to the stream or download the mp3."},
            "trade":{0:"This values the company, whose shares trade at 247p, at 16 times prospective profits.",
                    1:"Do you think she wants to trade with anyone"},
}

def main():
    model_type = 'bert'
    model_weights = 'bert-large-cased'
    tokenizer = BertTokenizer.from_pretrained(model_weights)
    model = BertModel.from_pretrained(model_weights)
    text_to_embedding = lambda text: text_to_embedding_fn(text, tokenizer, model)

    example_embeddings = {}
    for k, v in examples.items():
        if len(v) > 0:
            w = k
            wid = tokenizer.encode(w, add_special_tokens=False)[0]
            embedding = {}
            for key, text in examples[w].items():
                try:
                    target_location = tokenizer.encode(text.lower(), 
                                                    add_special_tokens=True).index(wid)
                except ValueError:
                    print(w)
                    print(text)
                outputs = text_to_embedding(text)
                embedding[key] = outputs[0][:,target_location,:].flatten().detach().cpu().numpy()
            example_embeddings[k] = embedding
    pickle.dump(example_embeddings, open('new_examples.pkl', 'wb'))
    

if __name__ == '__main__':
    main()
