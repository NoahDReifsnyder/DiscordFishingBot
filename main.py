'''
Discord Fishing Bot. Made with love <3
'''
import os
import pdb

import discord
from discord.ext import commands
from dotenv import load_dotenv
from database import Database
import inspect

import os
import uuid
import time
from datetime import timedelta

keep_delay = 24 * 60 * 60  # 24 hours in seconds
fish_delay = 15 * 60  # 15 minutes in seconds
pending_delay = 2 * 60  # 2 minutes in seconds
old_time = 365 * 24 * 60 * 60  # 1 year in seconds

IMAGE_DIR = "fish_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# Load the .env file
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Set up intents (what events the bot can see)
intents = discord.Intents.default()
intents.message_content = True# required to read message text
intents.members = True  # required to access member info
# Create the bot instance
bot = commands.Bot(command_prefix="!", intents=intents)
bot.help_command = None  # disable default help


@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")

    '''for guild in bot.guilds:
        # Prefer the system channel if it exists and is usable
        channel = guild.system_channel

        # Fallback: first text channel where the bot can send messages
        if channel is None:
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    channel = ch
                    break

        if channel is None:
            # No suitable channel, skip this guild
            continue

        try:
            await channel.send("🐟 FishBot is now online and ready to fish!")
        except Exception as e:
            print(f"Could not send startup message in {guild.name}: {e}")'''

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    print(f"📨 Message from {message.author}: {message.content}")

    # pass control back to the commands extension
    await bot.process_commands(message)


def get_file(db, fish_name):
    """Retrieve the image file for a given fish from the database."""
    file_name = db[fish_name]['file_name']
    file_path = os.path.join(IMAGE_DIR, file_name)
    return discord.File(file_path, filename=file_name)


@bot.command()
async def edit(ctx,
               name: str = commands.param(
                   description="The name of the fish type to edit."
               ),
               weight: float = commands.param(
                   default=None,
                   description="The new weight (score) of the fish type (<= 0)."
               ),
               rarity: float = commands.param(
                   default=None,
                   description="The new rarity of the fish type (>= 0). Higher rarity means less common."
               )
               ):
    """edits existing fish in the pond."""

    with Database("fish_types.db") as db:
        # check if name already exists, throw error if true
        if name not in db:
            await ctx.send(f"{name} does not exist in the pond!, cannot edit")
            return
        if weight is not None:
            if weight <= 0:
                await ctx.send("Weight must be a positive number!")
                return
            db[name]['weight'] = weight
        if rarity is not None:
            if rarity <= 0:
                await ctx.send("Rarity must be a positive number!")
                return
            db[name]['rarity'] = rarity
    await ctx.send(f"{name} has been edited in the pond!")

@bot.command()
async def fish(ctx):
    """Simulate fishing and return a random catch."""

    def _compute_probabilities(db):
        """
        fish_table: list of (name, rarity_value)
        returns: dict {name: probability_between_0_and_1}
        """
        total = sum(db[fish]["rarity"] for fish in db.keys())
        if total == 0:
            raise ValueError("Total rarity is 0; cannot normalize.")

        return {fish: 1 - (db[fish]['rarity'] / total) for fish in db.keys()}

    def _select_fish(db):
        import random
        rarity_probabilities = _compute_probabilities(db)
        total_rarity = sum(rarity_probabilities.values())
        pick = random.uniform(0, total_rarity)
        current = 0
        for fish in rarity_probabilities:
            current += rarity_probabilities[fish]
            if current >= pick:
                return fish
        return None  # Fallback, should not happen

    with Database("caught_fish.db") as caught_db:
        if ctx.author.id in caught_db:
            pending_info = caught_db[ctx.author.id].get("Pending", {})
            last_fish = caught_db[ctx.author.id].get("LastFish", time.time() - old_time)
            last_keep = caught_db[ctx.author.id].get("LastKeep", time.time() - old_time)
            pending_time = pending_info.get("time", time.time() - old_time)
            current_time = time.time()
            if current_time - pending_time > pending_delay:
                # auto release fish
                if pending_info:
                    caught_db[ctx.author.id].pop("Pending")
            if caught_db[ctx.author.id].get("Pending", False):
                fish = pending_info.get("fish")
                time_left = format_duration(pending_delay - int(current_time - pending_time))
                await ctx.send(f"{ctx.author.mention} You have a pending {fish} to deal with (auto release in {time_left} seconds)! Please use !keep or !release before fishing again.")
                return
            if current_time - last_keep < keep_delay:
                time_left = format_duration(keep_delay - (current_time - last_keep))
                await ctx.send(f"{ctx.author.mention} You need to wait 24 hrs after catching a fish to fish again. Time left: {time_left}")
                return
            if current_time - last_fish < fish_delay:
                time_left = format_duration(fish_delay - (current_time - last_fish))
                await ctx.send(f"{ctx.author.mention} You must wait {time_left} before fishing again!")
                return

    with Database("fish_types.db") as db:
        fish = _select_fish(db)
        if fish:
            with Database("caught_fish.db") as caught_db:
                if ctx.author.id not in caught_db:
                    caught_db[ctx.author.id] = {}
                if fish not in caught_db[ctx.author.id]:
                    caught_db[ctx.author.id][fish] = 0
                caught_db[ctx.author.id]["Pending"] = {"fish": fish, "time": ctx.message.created_at.timestamp()}
                caught_db[ctx.author.id]["LastFish"] = ctx.message.created_at.timestamp()
            file = get_file(db, fish)
            await ctx.send(f"{ctx.author.mention} You cast your line and caught {fish}. You have 2 minutes to !keep (!release to release early)", file=file)
        else:
            await ctx.send("You cast your line but caught nothing this time.")  ###this probably means pond has no fish
    return


@bot.command()
async def stock(ctx,
                name: str = commands.param(
                    description="The name of the fish type to add to the pond.",
                ),
                weight: float = commands.param(
                    description="The weight (score) of the fish type (<= 0)."
                ),
                rarity: float = commands.param(
                    description="The rarity of the fish type (>= 0). Higher rarity means less common."
                )
                ):
    """adds new fish to the pond."""

    if not ctx.message.attachments:
        await ctx.send("Please attach an image of the fish to this command.")
        return

    attachment = ctx.message.attachments[0]

    if attachment.content_type is not None and not attachment.content_type.startswith("image/"):
        await ctx.send("The attached file doesn't look like an image.")
        return
    # Pick an extension from the original filename
    _, ext = os.path.splitext(attachment.filename)
    if not ext:
        ext = ".png"  # fallback

    # Generate a safe unique filename
    safe_name = name.replace(" ", "_")
    file_name = f"{safe_name}_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(IMAGE_DIR, file_name)

    # Save file to disk
    await attachment.save(file_path)
    # check args are correct and send error if not
    print("name:", name)
    print("weight:", weight)
    print("rarity:", rarity)
    if weight <= 0:
        await ctx.send("Weight must be a positive number!")
        return
    print(rarity)
    if rarity <= 0:
        await ctx.send("Rarity must be a positive number!")
        return
    with Database("fish_types.db") as db:
        # check if name already exists, throw error if true
        if name in db:
            await ctx.send(f"A type of {name} already exists in the pond!, please extirpate it first")
            return
        db[name] = {
            "name": name,
            "weight": weight,
            "rarity": rarity,
            "file_name": file_name
        }
    await ctx.send(f"The pond has been stocked with {name}!")


@stock.error
async def stock_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Usage: !stock <name> <weight> <rarity>")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Weight and rarity must be numbers!")


@bot.command()
async def extirpate(ctx,
                    name: str = commands.param(
                        description="The name of the fish type to remove from the pond."
                    )
                    ):
    """removes fish from the pond."""

    with Database("fish_types.db") as db:
        # check if name already exists, throw error if true
        if name not in db:
            await ctx.send(f"{name} does not exist in the pond!, cannot extirpate")
            return
        del db[name]
    await ctx.send(f"{name} has been extirpated from the pond!")


@bot.command()
async def pond(ctx):
    """lists all fish in the pond"""
    message = f"{ctx.author.mention}\n**🐟 Fish Types in the Pond 🐟**\n"
    with Database("fish_types.db") as db:
        if len(db) == 0:
            await ctx.send("The pond is currently empty.")
            return
        for fish in db:
            description = f"Weight: {db[fish]['weight']}, Rarity: {db[fish]['rarity']}"
            message += f"- {fish}: {description}\n"
        await ctx.send(message)
    return

def make_fish_embed(db, fish):

    # Base embed with name + rarity
    rarity = db[fish]['rarity']
    weight = db[fish]['weight']
    file = get_file(db, fish)
    embed = discord.Embed(
        title=fish,
        description=f"Weight: `{weight}`\nRarity: `{rarity}`",
    )
    image_file = db[fish]['file_name']


    embed.set_image(url=f"attachment://{image_file}")

    return embed, file


@bot.command()
async def storage(ctx,
                   user: discord.Member = commands.param(
                       default=None,
                       description=":   The user whose storage to show. Defaults to you."
                   ),
                   ):
    """Displays a users caught fish to them, or a provided users fish."""
    with Database("caught_fish.db") as db:
        if ctx.author.id not in db or len(db[ctx.author.id]) == 0:
            await ctx.send("You haven't caught any fish yet!")
            return
        caught_fish = db[ctx.author.id]
        message = f"{ctx.author.mention}, here are the fish you have caught:\n"
        for fish in caught_fish:
            count = caught_fish[fish]
            with Database("fish_types.db") as fish_db:
                if fish in fish_db:
                    message += f"{count} x {fish} (Weight: {fish_db[fish]['weight']}, Rarity: {fish_db[fish]['rarity']})\n"

        await ctx.send(message)
    return

@bot.command()
async def aquarium(ctx,
                   user: discord.Member = commands.param(
                       default=None,
                       description="The user whose aquarium to show. Defaults to you."
                   ),
                   ):
    """Displays a users caught fish to them, or a provided users fish."""
    target_user = user or ctx.author

    with Database("caught_fish.db") as db:
        if target_user.id not in db or len(db[target_user.id]) == 0:
            await ctx.send(f"{target_user.mention} hasn't caught any fish yet!")
            return
        caught_fish = db[target_user.id]
        message = f"{target_user.mention}'s aquarium:\n"
        message2 = f"{target_user.mention}'s aquarium:\n"
        files = []
        embeds = []
        #get up to the 5 rarest fish in target_users storage
        with Database("fish_types.db") as fish_db:
            sorted_fish = sorted(caught_fish.items(), key=lambda item: fish_db[item[0]]['rarity'] if item[0] in fish_db else float('inf'), reverse=True)
            for fish, count in sorted_fish[:3]:
                embed, file = make_fish_embed(fish_db, fish)
                embeds.append(embed)
                files.append(file)
                message2 += f"{count} x {fish} (Rarity: {fish_db[fish]['rarity']})\n"

        await ctx.send(message2, files=files)
    return

@bot.command()
async def info(ctx,
                 fish: str = commands.param(
                     description="The name of the fish type to search for in the pond."
                 )
                 ):
    """searches for fish in the pond."""

    with Database("fish_types.db") as db:
        # check if name already exists, throw error if true
        if fish not in db:
            await ctx.send(f"{fish} does not exist in the pond!")
            return
        embed, file = make_fish_embed(db, fish)
        #create leaderboard style embed for users who have caught the most of this fish, as well as authors number of caught fish
        embed = discord.Embed(
            title="Leaderboard",
            description=f"Top catchers of {fish}:",
        )
        with Database("caught_fish.db") as caught_db:
            leaderboard = []
            for user_id in caught_db:
                user_caught = caught_db[user_id]
                if fish in user_caught:
                    leaderboard.append((int(user_id), user_caught[fish]))
            #sort leaderboard by number of fish caught
            leaderboard.sort(key=lambda x: x[1], reverse=True)
            #take top 3, but is safe if less than 3 exist
            top_3 = leaderboard
            if len(leaderboard) >=3:
                top_3 = leaderboard[:3]
            for rank, (user_id, count) in enumerate(top_3, start=1):
                user = ctx.guild.get_member(user_id)
                if user:
                    embed.add_field(name=f"{rank}. {user.display_name}", value=f"Caught: {count}", inline=False)
            #find author's rank
            author_rank = None
            for rank, (user_id, count) in enumerate(leaderboard, start=1):
                if user_id == ctx.author.id:
                    author_rank = rank
                    author_count = count
                    break
            if author_rank and author_rank > len(top_3):
                embed.set_footer(text=f"Your rank: {author_rank} with {author_count} caught.")
            elif not author_rank:
                embed.set_footer(text="You have not caught this fish yet.")
        await ctx.send(f"{ctx.author.mention}", embed=embed, file=file)
    return

@bot.command()
async def keep(ctx):
    """Places your current Pending fish into storage"""
    with Database("caught_fish.db") as caught_db:
        if ctx.author.id not in caught_db:
            await ctx.send(f"{ctx.author.mention} You have no pending fish.")
            return
        pending_info = caught_db[ctx.author.id].get("Pending", {})
        pending_time = pending_info.get("time", time.time() - (3 * 60))
        current_time = time.time()
        if current_time - pending_time > 120:
            # auto release fish
            caught_db[ctx.author.id].pop("Pending")
            await ctx.send(f"{ctx.author.mention} Your pending fish has expired and was released.")
            return
        if not caught_db[ctx.author.id].get("Pending", False):
            await ctx.send(f"{ctx.author.mention} You have no pending fish to keep.")
            return
        fish = pending_info.get("fish")
        caught_db[ctx.author.id].pop("Pending")
        caught_db[ctx.author.id][fish] = caught_db[ctx.author.id].get(fish, 0) + 1
        caught_db[ctx.author.id]["LastKeep"] = ctx.message.created_at.timestamp()

        await ctx.send(f"{ctx.author.mention} You have kept your {fish}! You now have {caught_db[ctx.author.id][fish]} of them in storage.")


@bot.command()
async def release(ctx):
    """Releases your current Pending fish"""
    with Database("caught_fish.db") as caught_db:
        if ctx.author.id not in caught_db:
            await ctx.send(f"{ctx.author.mention} You have no pending fish.")
            return
        pending_info = caught_db[ctx.author.id].get("Pending", {})
        pending_time = pending_info.get("time", time.time() - (3 * 60))
        current_time = time.time()
        if current_time - pending_time > 120:
            # auto release fish
            caught_db[ctx.author.id].pop("Pending")
            await ctx.send(f"{ctx.author.mention} Your pending fish has expired and was released.")
            return
        if not caught_db[ctx.author.id].get("Pending", False):
            await ctx.send(f"{ctx.author.mention} You have no pending fish to release.")
            return
        fish = pending_info.get("fish")
        caught_db[ctx.author.id].pop("Pending")

        await ctx.send(f"{ctx.author.mention} You have released your {fish}. Better luck next time!")

@bot.command(name="help")
async def custom_help(ctx,
                      command_or_flag: str = commands.param(
                          default=None,
                          description="The command to show help for, or -v/--verbose for detailed list."
                      ),
                      ):
    """
   Show help for all commands, or a specific command.
   """

    # ---------- parse tokens ----------
    verbose_global = False


    verbose_global = False
    command_name = None

    if command_or_flag in ("-v", "--verbose"):
        verbose_global = True
    else:
        command_name = command_or_flag
        pass


    # ---------- helpers ----------
    def get_description(cmd: commands.Command) -> str:
        """First line of the command's docstring, or fallback."""
        doc = inspect.getdoc(cmd.callback) or ""
        return doc.splitlines()[0] if doc else "No description."


    def format_params(cmd: commands.Command) -> list[str]:
        """Format parameters using commands.Param / commands.parameter metadata."""
        lines: list[str] = []
        for name, param in cmd.params.items():
            if name == "ctx":
                continue

            desc = getattr(param, "description", None) or "No description."
            required = param.default is param.empty
            opt_text = "" if required else " (optional)"

            annotation = param.annotation
            if annotation is param.empty:
                type_name = "Any"
            else:
                type_name = getattr(annotation, "__name__", str(annotation))

            # one nice markdown bullet line
            lines.append(f"- {name} ({type_name}{opt_text}) — {desc}")
        return lines


    # ---------- no specific command: list mode ----------
    if command_name is None:
        lines: list[str] = []
        if verbose_global:
            lines.append("**Commands (verbose)**")
        else:
            lines.append("**Commands**")
        lines.append(
            "Use !help -v for detailed argument info, or !help <command> for details on a single command."
        )
        lines.append("")

        for cmd in bot.commands:
            if cmd.hidden:
                continue

            desc = get_description(cmd)
            lines.append(f"!{cmd.qualified_name} — {desc}")

            if verbose_global:
                params = format_params(cmd)
                if params:
                    # indent the argument bullets visually under the command
                    for p in params:
                        lines.append(f"  {p}")
            lines.append("")  # blank line between commands

        boxed = "\n".join(f"> {line.strip()}" for line in lines[:-1])
        print(boxed)
        await ctx.send(boxed)
        return

    # ---------- specific command: always verbose ----------
    cmd = bot.get_command(command_name)
    if cmd is None:
        await ctx.send(f"Unknown command: `{command_name}`")
        return

    desc = get_description(cmd)
    usage = f"!{cmd.qualified_name} {cmd.signature or ''}".strip()

    lines: list[str] = []
    lines.append(f"**!{cmd.qualified_name}**")
    lines.append("")
    lines.append(f"**Usage:** `{usage}`")
    lines.append("")
    lines.append(desc)
    lines.append("")
    lines.append("**Arguments:**")

    params = format_params(cmd)
    if params:
        lines.extend(params)
    else:
        lines.append("- (none)")

    boxed = "\n".join(f"> {line.strip()}" for line in lines)
    print(boxed)
    await ctx.send(boxed)


def format_duration(seconds: int) -> str:
    parts = []

    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, rem = divmod(rem, 60)
    secs = rem

    if days:
        parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    if secs and not parts:  # only show seconds if nothing else
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    print(parts)
    test = " ".join(parts) if parts else "0 seconds"
    print(test)
    return " ".join(parts) if parts else "0 seconds"


@bot.command()
async def leaderboard(ctx):
    """Displays the top fishers."""
    #metric is total weight
    with Database("caught_fish.db") as caught_db:
        scores = []
        author_rank = None
        author_score = 0
        with Database("fish_types.db") as fish_db:
            for user_id in caught_db:
                user_id = int(user_id)
                total_weight = 0
                for fish in caught_db[user_id]:
                    if fish in fish_db:
                        total_weight += fish_db[fish]['weight'] * caught_db[user_id][fish]
                scores.append((int(user_id), total_weight))
                if user_id == ctx.author.id:
                    author_score = total_weight
        #sort scores by total weight
        scores.sort(key=lambda x: x[1], reverse=True)
        author_rank = scores.index((ctx.author.id, author_score)) + 1 if (ctx.author.id, author_score) in scores else None
        #take top 5
        top_5 = scores
        if len(scores) >=5:
            top_5 = scores[:5]
        message = f"{ctx.author.mention}\n"
        message += "\n**🏆 Fishing Leaderboard 🏆**\n"
        for rank, (user_id, score) in enumerate(top_5, start=1):
            user = ctx.guild.get_member(user_id)
            if user:
                message += f"{rank}. {user.display_name} - Total Weight: {score}\n"
        #show the user and their rank if they aren't in top_5
        if author_rank and author_rank > 5:
            message += f"...\n{author_rank}. {ctx.author.display_name} - Total Weight: {author_score}\n"
        await ctx.send(message)

@bot.command()
async def rules(ctx):
    """Explains the rules of the fishing game."""
    # use the !fish command to catch a fish
    # you have 2 minutes to decide if you want to keep the fish
    # when you keep a fish, you must wait 24 hours to fish again
    # you must also wait 15 minutes between fishing attempts,
    # even after releasing your pending fish


    keep_delay_str = format_duration(keep_delay)
    fish_delay_str = format_duration(fish_delay)
    pending_delay_str = format_duration(pending_delay)
    # Build the lines first
    lines = [
        "**🎣 Fishing Game Rules 🎣**",
        " ",
        "Here are the rules of the game:",
        " ",
        "1. Use `!fish` to cast your line and catch a random fish.",
        (
            f"2. After you catch a fish, you have **{pending_delay_str}** "
            "to decide whether to add the fish to your storage or release it."
        ),
        (
            f"3. If you **keep** the fish, you must wait **{keep_delay_str}** "
            "before you can use `!fish` again."
        ),
        (
            f"4. Even if you **release** the fish, you must wait **{fish_delay_str}** "
            "between fishing attempts."
        ),
        "5. Plan your catches wisely and build the best aquarium you can! 🐟",
    ]

    # Turn it into a blockquote: each non-empty line prefixed with "> "
    # blank lines become just ">" to keep spacing inside the quote
    blockquoted = "\n".join("> " + line if line else ">" for line in lines)

    await ctx.send(blockquoted)

@bot.command()
async def clear_timers(ctx):
    """Clears your fishing timers (for testing purposes)."""
    with Database("caught_fish.db") as caught_db:
        if ctx.author.id in caught_db:
            caught_db[ctx.author.id]["LastFish"] = old_time
            caught_db[ctx.author.id]["LastKeep"] = old_time
            if "Pending" in caught_db[ctx.author.id]:
                caught_db[ctx.author.id].pop("Pending")
    await ctx.send(f"{ctx.author.mention} Your fishing timers have been cleared.")

# Start the bot
bot.run(TOKEN)
