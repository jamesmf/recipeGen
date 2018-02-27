# recipeGen
Natural Language Generation playground with recipes

## Data
The data used to train the NLG models in this repository comes from Wikibooks, and shares the same license:
https://en.wikibooks.org/wiki/Wikibooks:GNU_Free_Documentation_License

The recipes have been reformatted for standard parsing. They terminate with $$$$, which helps for generation.

## Models
This codebase has been through a few major refactors, and its most recent iteration uses a character-level CNN. The model expects inputs for the last `n1` characters and the previous `n2` characters (overlapping) and processes them using shared layers. It then predicts the next `n3` characters

The newest version of the model will also feature a discriminator built-in. The hope with adding the discriminator is that it will learn to recognize that repeating ingredients or adding two ingredients that don't go together is globally not optimal.

## Examples
Without the discriminator, here's an example recipe, seeded with the name "wedding cake"
```
name:
wedding cake

ingredients:

1 c. milk
1/2 c. butter
1 tsp. vanilla
1/2 tsp. salt
1/2 c. butter
1/2 c. grated parmesan cheese
1 c. sugar
1/4 c. corn green or sweet potatoes

directions:
mix all ingredients and stir into a 9 x 13-inch baking dish. bake at 400 degrees for 10 minutes or until cookies sliced stock. cover and simmer 15 minutes or until sauce is the consistency. roast mixture to cover the top of bottom of a slice bowl. spoon on top. bake at 350 degrees at 350 degrees or onion. sprinkle with cheese and serve.
```
